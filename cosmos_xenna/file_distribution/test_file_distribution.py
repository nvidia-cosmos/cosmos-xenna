# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the distributed file download system.

Run with:

RAY_DEDUP_LOGS=0 RUST_BACKTRACE=full RUST_LOG=debug uv run pytest cosmos_xenna/file_distribution/test_file_distribution.py -s --tb=native
"""  # noqa: E501

import os
import pathlib
import shutil
import time
import zipfile

os.environ["RAY_DEDUP_LOGS"] = "0"

import attrs
import obstore as obs
import pytest
import ray

from cosmos_xenna.file_distribution import (
    DownloadRequest,
    ObjectDownloadRequest,
    ObjectStoreConfig,
    PrefixDownloadRequest,
    SingleNodeTestingInfo,
    UnpackMethod,
    UnpackOptions,
    download_distributed,
)
from cosmos_xenna.file_distribution._common import get_cache_path_for_directory, get_cache_path_for_file, resolve_path

_OBSTORE_PATH = pathlib.Path("/tmp/pretend_object_store")


@attrs.define
class TestS3Object:
    key: str
    bytes: bytes


@pytest.fixture
def testing_info():
    """Create single-node testing configuration."""
    return SingleNodeTestingInfo(num_fake_nodes=3)


@pytest.fixture(autouse=True)
def run_around_tests():
    """Setup and teardown for each test."""
    # Setup: Clean up any existing test data
    test_storage_root = pathlib.Path("/tmp/p2p_download_test")
    if test_storage_root.exists():
        shutil.rmtree(test_storage_root)
    if os.path.exists(_OBSTORE_PATH):
        shutil.rmtree(_OBSTORE_PATH)
    _OBSTORE_PATH.mkdir(parents=True, exist_ok=True)

    ray.init()

    yield

    if ray.is_initialized():
        ray.shutdown()

    # Teardown: Clean up test data
    if test_storage_root.exists():
        shutil.rmtree(test_storage_root)
    shutil.rmtree(_OBSTORE_PATH)


def make_client_config_and_upload_objects(test_objects: list[TestS3Object]) -> ObjectStoreConfig:
    config = ObjectStoreConfig(uri="file://" + _OBSTORE_PATH.as_posix())
    obstore = obs.store.from_url(config.uri)
    for obj in test_objects:
        obstore.put(obj.key, obj.bytes)
        assert obstore.get(obj.key).bytes() == obj.bytes
    return config


class TestDistributedDownload:
    """Test cases for the distributed download system."""

    def test_single_file_download(self, testing_info: SingleNodeTestingInfo):
        """Test downloading a single small file."""

        # Create test objects dynamically
        test_objects = [
            TestS3Object(
                key="test-bucket/small-file.txt",
                bytes=b"Hello, World!",
            )
        ]
        test_client_factory = make_client_config_and_upload_objects(test_objects)

        dest_path = pathlib.Path("/tmp/test_output/small-file.txt")

        requests = [
            DownloadRequest(
                value=ObjectDownloadRequest(
                    uri="test-bucket/small-file.txt",
                    destination=dest_path,
                )
            )
        ]

        download_distributed(requests, test_client_factory, testing_info=testing_info, node_parallelism=10)

        for node_id in range(testing_info.num_fake_nodes):
            expected_path = resolve_path(dest_path, str(node_id), is_test=True)
            assert expected_path.exists()
            assert expected_path.read_bytes() == b"Hello, World!"

            # Verify cache file was created
            cache_path = get_cache_path_for_file(expected_path)
            assert cache_path.exists()

    def test_small_chunked_file_download(self, testing_info: SingleNodeTestingInfo):
        """Test downloading a single small file that gets chunked."""

        # Create test objects dynamically
        test_objects = [
            TestS3Object(
                key="test-bucket/small-file.txt",
                bytes=b"Hello, World!",
            )
        ]
        test_client_factory = make_client_config_and_upload_objects(test_objects)

        dest_path = pathlib.Path("/tmp/test_output/small-file.txt")

        requests = [
            DownloadRequest(
                value=ObjectDownloadRequest(
                    uri="test-bucket/small-file.txt",
                    destination=dest_path,
                )
            )
        ]

        download_distributed(
            requests, test_client_factory, testing_info=testing_info, node_parallelism=10, chunk_size_bytes=4
        )

        for node_id in range(testing_info.num_fake_nodes):
            expected_path = resolve_path(dest_path, str(node_id), is_test=True)
            assert expected_path.exists()
            assert expected_path.read_bytes() == b"Hello, World!"

            # Verify cache file was created
            cache_path = get_cache_path_for_file(expected_path)
            assert cache_path.exists()

    def test_chunked_file_download(self, testing_info: SingleNodeTestingInfo):
        """Test downloading a large file that gets chunked."""

        # Create a large test object (200MB)
        large_content = b"X" * (200 * 1024 * 1024)
        test_objects = [
            TestS3Object(
                key="test-bucket/large-file.dat",
                bytes=large_content,
            )
        ]
        test_client_factory = make_client_config_and_upload_objects(test_objects)

        dest_path = pathlib.Path("/tmp/test_output/large-file.dat")

        requests = [
            DownloadRequest(
                value=ObjectDownloadRequest(
                    uri="test-bucket/large-file.dat",
                    destination=dest_path,
                )
            )
        ]

        download_distributed(
            requests,
            test_client_factory,
            chunk_size_bytes=50 * 1024 * 1024,
            testing_info=testing_info,
            node_parallelism=10,
        )

        for node_id in range(testing_info.num_fake_nodes):
            expected_path = resolve_path(dest_path, str(node_id), is_test=True)
            assert expected_path.exists()
            assert len(expected_path.read_bytes()) == 200 * 1024 * 1024
            assert expected_path.read_bytes() == b"X" * (200 * 1024 * 1024)

            # Verify cache file was created
            cache_path = get_cache_path_for_file(expected_path)
            assert cache_path.exists()

    def test_prefix_download(self, testing_info: SingleNodeTestingInfo):
        """Test downloading all files under a prefix."""

        # Create test objects for prefix download
        test_objects = [
            TestS3Object(
                key="test-bucket/dataset/train/data1.txt",
                bytes=b"Training data 1",
            ),
            TestS3Object(
                key="test-bucket/dataset/train/data2.txt",
                bytes=b"Training data 2",
            ),
            TestS3Object(
                key="test-bucket/dataset/test/data3.txt",
                bytes=b"Test data 3",
            ),
        ]
        test_client_factory = make_client_config_and_upload_objects(test_objects)

        dest_dir = pathlib.Path("/tmp/test_output/dataset")

        requests = [
            DownloadRequest(
                value=PrefixDownloadRequest(
                    uri="test-bucket/dataset/",
                    destination=dest_dir,
                )
            )
        ]

        download_distributed(requests, test_client_factory, testing_info=testing_info, node_parallelism=10)

        for node_id in range(testing_info.num_fake_nodes):
            # Verify all files were downloaded with correct structure
            expected_dest_dir = resolve_path(dest_dir, str(node_id), is_test=True)
            train_data1 = expected_dest_dir / "train/data1.txt"
            train_data2 = expected_dest_dir / "train/data2.txt"
            test_data3 = expected_dest_dir / "test/data3.txt"

            assert train_data1.exists()
            assert train_data2.exists()
            assert test_data3.exists()

            assert train_data1.read_bytes() == b"Training data 1"
            assert train_data2.read_bytes() == b"Training data 2"
            assert test_data3.read_bytes() == b"Test data 3"

            # Verify cache files were created
            for file_path in [train_data1, train_data2, test_data3]:
                cache_path = get_cache_path_for_file(file_path)
                assert cache_path.exists()

    def test_caching_behavior(self, testing_info: SingleNodeTestingInfo):
        """Test that caching works correctly and avoids redundant downloads."""

        # Create test object for caching test
        test_objects = [
            TestS3Object(
                key="test-bucket/small-file.txt",
                bytes=b"Hello, World!",
            )
        ]
        test_client_factory = make_client_config_and_upload_objects(test_objects)

        dest_path = pathlib.Path("/tmp/test_output/cached-file.txt")

        # First download
        requests = [
            DownloadRequest(
                value=ObjectDownloadRequest(
                    uri="test-bucket/small-file.txt",
                    destination=dest_path,
                )
            )
        ]

        download_distributed(requests, test_client_factory, testing_info=testing_info, node_parallelism=10)

        # Verify file and cache exist for each node
        for node_id in range(testing_info.num_fake_nodes):
            expected_path = resolve_path(dest_path, str(node_id), is_test=True)
            assert expected_path.exists()
            cache_path = get_cache_path_for_file(expected_path)
            assert cache_path.exists()
            assert expected_path.read_bytes() == b"Hello, World!"

        # Second download should use cache - cache files should still exist
        download_distributed(requests, test_client_factory, testing_info=testing_info, node_parallelism=10)

        # File should still exist and cache should still be present
        for node_id in range(testing_info.num_fake_nodes):
            expected_path = resolve_path(dest_path, str(node_id), is_test=True)
            assert expected_path.exists()
            cache_path = get_cache_path_for_file(expected_path)
            assert cache_path.exists()
            assert expected_path.read_bytes() == b"Hello, World!"

    def test_unpack_zip_file(self, testing_info: SingleNodeTestingInfo):
        """Test downloading and unpacking a ZIP file."""

        # Create test ZIP object
        zip_content = self._create_test_zip()
        test_objects = [
            TestS3Object(
                key="test-bucket/archive.zip",
                bytes=zip_content,
            )
        ]
        test_client_factory = make_client_config_and_upload_objects(test_objects)

        archive_path = pathlib.Path("/tmp/test_output/archive.zip")
        unpack_path = pathlib.Path("/tmp/test_output/unpacked")

        requests = [
            DownloadRequest(
                value=ObjectDownloadRequest(
                    uri="test-bucket/archive.zip",
                    destination=archive_path,
                    unpack_options=UnpackOptions(unpack_destination=unpack_path, unpack_method=UnpackMethod.ZIP),
                )
            )
        ]

        download_distributed(requests, test_client_factory, testing_info=testing_info, node_parallelism=10)

        for node_id in range(testing_info.num_fake_nodes):
            # Verify archive was downloaded
            expected_archive_path = resolve_path(archive_path, str(node_id), is_test=True)
            expected_unpack_path = resolve_path(unpack_path, str(node_id), is_test=True)

            assert expected_archive_path.exists()

            # Verify archive was unpacked
            assert expected_unpack_path.exists()
            assert (expected_unpack_path / "file1.txt").exists()
            assert (expected_unpack_path / "file2.txt").exists()
            assert (expected_unpack_path / "file1.txt").read_text() == "Content 1"
            assert (expected_unpack_path / "file2.txt").read_text() == "Content 2"

            # Verify cache files for both archive and unpacked directory
            archive_cache = get_cache_path_for_file(expected_archive_path)
            unpack_cache = get_cache_path_for_directory(expected_unpack_path)
            assert archive_cache.exists()
            assert unpack_cache.exists()

    def _create_test_zip(self) -> bytes:
        """Create a test ZIP file with some content."""
        import io

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("file1.txt", "Content 1")
            zip_file.writestr("file2.txt", "Content 2")
        return zip_buffer.getvalue()

    def test_multiple_files_distribution(self, testing_info: SingleNodeTestingInfo):
        """Test that multiple files are distributed across nodes effectively."""

        # Create test objects for multiple files distribution
        large_content = b"X" * (200 * 1024 * 1024)  # 200MB file
        test_objects = [
            TestS3Object(
                key="test-bucket/small-file.txt",
                bytes=b"Hello, World!",
            ),
            TestS3Object(
                key="test-bucket/large-file.dat",
                bytes=large_content,
            ),
        ]
        test_client_factory = make_client_config_and_upload_objects(test_objects)

        # Download multiple files to test P2P distribution
        files = [
            ("test-bucket/small-file.txt", "/tmp/test_output/file1.txt"),
            ("test-bucket/large-file.dat", "/tmp/test_output/file2.dat"),
        ]

        requests = []
        for uri, dest in files:
            dest_path = pathlib.Path(dest)
            requests.append(
                DownloadRequest(
                    value=ObjectDownloadRequest(
                        uri=uri,
                        destination=dest_path,
                    )
                )
            )

        download_distributed(requests, test_client_factory, testing_info=testing_info, node_parallelism=10)

        # Verify all files were downloaded at each node
        for node_id in range(testing_info.num_fake_nodes):
            for uri, dest in files:
                dest_path = pathlib.Path(dest)
                expected_path = resolve_path(dest_path, str(node_id), is_test=True)
                assert expected_path.exists()

                # Verify content matches expected
                if uri == "test-bucket/small-file.txt":
                    assert expected_path.read_bytes() == b"Hello, World!"
                elif uri == "test-bucket/large-file.dat":
                    assert len(expected_path.read_bytes()) == 200 * 1024 * 1024
                    assert expected_path.read_bytes() == b"X" * (200 * 1024 * 1024)

                # Verify cache files were created
                cache_path = get_cache_path_for_file(expected_path)
                assert cache_path.exists()

    def test_heavy_load_balancing(self, testing_info: SingleNodeTestingInfo):
        """Test scheduler with significantly more files than nodes to stress load balancing."""

        # Create test objects and requests
        test_objects = []
        requests = []

        # Create many more files than we have nodes (should be 3 nodes by default)
        num_files = 20

        # Mix of regular files, archives to unpack, and prefix downloads
        for i in range(num_files):
            if i % 4 == 0:  # Every 4th file is an archive
                dest_path = pathlib.Path(f"/tmp/test_output/archive_{i}.zip")
                unpack_dest = pathlib.Path(f"/tmp/test_output/unpacked_{i}")
                requests.append(
                    DownloadRequest(
                        value=ObjectDownloadRequest(
                            uri=f"test-bucket/archive_{i}.zip",
                            destination=dest_path,
                            unpack_options=UnpackOptions(unpack_dest, UnpackMethod.ZIP),
                        )
                    )
                )
                # Add the archive to test objects
                test_objects.append(
                    TestS3Object(
                        key=f"test-bucket/archive_{i}.zip",
                        bytes=self._create_test_zip(),
                    )
                )
            elif i % 7 == 0:  # Some files are in prefixes
                dest_path = pathlib.Path(f"/tmp/test_output/dataset_{i}")
                requests.append(
                    DownloadRequest(
                        value=PrefixDownloadRequest(
                            uri=f"test-bucket/dataset_{i}/",
                            destination=dest_path,
                        )
                    )
                )
                # Add some files in the prefix
                for j in range(3):
                    test_objects.append(
                        TestS3Object(
                            key=f"test-bucket/dataset_{i}/file_{j}.txt",
                            bytes=f"Dataset {i} file {j}".encode(),
                        )
                    )
            else:  # Regular files
                dest_path = pathlib.Path(f"/tmp/test_output/file_{i}.txt")
                requests.append(
                    DownloadRequest(
                        value=ObjectDownloadRequest(
                            uri=f"test-bucket/file_{i}.txt",
                            destination=dest_path,
                        )
                    )
                )
                test_objects.append(
                    TestS3Object(
                        key=f"test-bucket/file_{i}.txt",
                        bytes=f"Content of file {i}".encode(),
                    )
                )

        test_client_factory = make_client_config_and_upload_objects(test_objects)
        download_distributed(requests, test_client_factory, testing_info=testing_info, node_parallelism=10)

        # Verify all files are properly distributed
        for node_id in range(testing_info.num_fake_nodes):
            node_root = pathlib.Path(f"/tmp/p2p_download_test/{node_id}")
            assert node_root.exists()
            # At least verify some files exist (full verification would be complex due to mixed types)
            file_count = len(list(node_root.rglob("*.txt"))) + len(list(node_root.rglob("*.zip")))
            assert file_count > 5  # Should have downloaded several files

    @pytest.mark.skip(reason="This test is flaky and needs to be fixed")
    def test_cache_corruption_scenarios(self, testing_info: SingleNodeTestingInfo):
        """Test various cache corruption and invalidation scenarios."""

        # Create test objects for cache corruption test
        test_objects = [
            TestS3Object(
                key="test-bucket/cache_test.txt",
                bytes=b"Hello, World!",
            ),
            TestS3Object(
                key="test-bucket/cache_test.zip",
                bytes=self._create_test_zip(),
            ),
        ]
        test_client_factory = make_client_config_and_upload_objects(test_objects)

        dest_path = pathlib.Path("/tmp/test_output/cache_test.txt")
        unpack_dest = pathlib.Path("/tmp/test_output/unpacked_cache")

        requests = [
            DownloadRequest(
                value=ObjectDownloadRequest(
                    uri="test-bucket/cache_test.txt",
                    destination=dest_path,
                )
            ),
            DownloadRequest(
                value=ObjectDownloadRequest(
                    uri="test-bucket/cache_test.zip",
                    destination=pathlib.Path("/tmp/test_output/cache_test.zip"),
                    unpack_options=UnpackOptions(unpack_dest, UnpackMethod.ZIP),
                )
            ),
        ]

        # First download - should create cache
        download_distributed(requests, test_client_factory, testing_info=testing_info)

        # Verify cache files exist
        for node_id in range(testing_info.num_fake_nodes):
            file_path = resolve_path(dest_path, str(node_id), is_test=True)
            cache_path = get_cache_path_for_file(file_path)
            assert file_path.exists()
            assert cache_path.exists()

            unpack_path = resolve_path(unpack_dest, str(node_id), is_test=True)
            unpack_cache_path = get_cache_path_for_directory(unpack_path)
            assert unpack_path.exists()
            assert unpack_cache_path.exists()

        # Scenario 1: Delete just the cache info file (should re-download)
        node_0_file = resolve_path(dest_path, "0", is_test=True)
        node_0_cache = get_cache_path_for_file(node_0_file)
        node_0_cache.unlink()

        # Scenario 2: Delete just the unpacked directory (should re-unpack)
        node_1_unpack = resolve_path(unpack_dest, "1", is_test=True)
        shutil.rmtree(node_1_unpack)

        # Scenario 3: Delete the original file but keep unpacked (should handle gracefully)
        node_2_zip = resolve_path(pathlib.Path("/tmp/test_output/cache_test.zip"), "2", is_test=True)
        node_2_zip.unlink()

        time.sleep(0.25)  # wait for the filesystem to catch up. Without this, this test will fail sometimes.
        # Run download again - should handle all corruption scenarios
        download_distributed(requests, test_client_factory, testing_info=testing_info)

        # Verify everything is properly restored
        for node_id in range(testing_info.num_fake_nodes):
            file_path = resolve_path(dest_path, str(node_id), is_test=True)
            cache_path = get_cache_path_for_file(file_path)
            assert file_path.exists()
            assert cache_path.exists()

            unpack_path = resolve_path(unpack_dest, str(node_id), is_test=True)
            unpack_cache_path = get_cache_path_for_directory(unpack_path)
            assert unpack_path.exists()
            assert unpack_cache_path.exists()

    def test_mixed_file_sizes_and_chunking(self, testing_info: SingleNodeTestingInfo):
        """Test with mix of small files (no chunking) and large files (chunking)."""

        # Create test objects for mixed file sizes
        small_content = b"Small file content"
        large_content = b"x" * 5000  # 5KB file, will create 5 chunks with 1KB chunks
        medium_content = b"y" * 1500  # 1.5KB file

        test_objects = [
            TestS3Object(
                key="test-bucket/small.txt",
                bytes=small_content,
            ),
            TestS3Object(
                key="test-bucket/large.txt",
                bytes=large_content,
            ),
            TestS3Object(
                key="test-bucket/medium.txt",
                bytes=medium_content,
            ),
        ]
        test_client_factory = make_client_config_and_upload_objects(test_objects)

        requests = [
            DownloadRequest(
                value=ObjectDownloadRequest(
                    uri="test-bucket/small.txt",
                    destination=pathlib.Path("/tmp/test_output/small.txt"),
                )
            ),
            DownloadRequest(
                value=ObjectDownloadRequest(
                    uri="test-bucket/large.txt",
                    destination=pathlib.Path("/tmp/test_output/large.txt"),
                )
            ),
            DownloadRequest(
                value=ObjectDownloadRequest(
                    uri="test-bucket/medium.txt",
                    destination=pathlib.Path("/tmp/test_output/medium.txt"),
                )
            ),
        ]

        # Use small chunk size to force chunking of the large file
        download_distributed(requests, test_client_factory, chunk_size_bytes=1024, testing_info=testing_info)

        # Verify all files downloaded correctly on all nodes
        for node_id in range(testing_info.num_fake_nodes):
            small_path = resolve_path(pathlib.Path("/tmp/test_output/small.txt"), str(node_id), is_test=True)
            large_path = resolve_path(pathlib.Path("/tmp/test_output/large.txt"), str(node_id), is_test=True)
            medium_path = resolve_path(pathlib.Path("/tmp/test_output/medium.txt"), str(node_id), is_test=True)

            assert small_path.exists()
            assert large_path.exists()
            assert medium_path.exists()

            assert small_path.read_bytes() == small_content
            assert large_path.read_bytes() == large_content
            assert medium_path.read_bytes() == medium_content

    def test_cache_validation_with_updated_files(self, testing_info: SingleNodeTestingInfo):
        """Test cache behavior when files change in the object store."""

        dest_path = pathlib.Path("/tmp/test_output/changing_file.txt")
        original_content = b"Original content"
        updated_content = b"Updated content after change"

        # Create initial test objects
        initial_objects = [
            TestS3Object(
                key="test-bucket/changing_file.txt",
                bytes=original_content,
            )
        ]
        test_client_factory = make_client_config_and_upload_objects(initial_objects)

        requests = [
            DownloadRequest(
                value=ObjectDownloadRequest(
                    uri="test-bucket/changing_file.txt",
                    destination=dest_path,
                )
            )
        ]

        # First download
        download_distributed(requests, test_client_factory, testing_info=testing_info)

        # Verify file downloaded
        for node_id in range(testing_info.num_fake_nodes):
            file_path = resolve_path(dest_path, str(node_id), is_test=True)
            assert file_path.exists()
            assert file_path.read_bytes() == original_content

        # Manually delete cache files to simulate cache invalidation
        for node_id in range(testing_info.num_fake_nodes):
            file_path = resolve_path(dest_path, str(node_id), is_test=True)
            cache_path = get_cache_path_for_file(file_path)
            if cache_path.exists():
                cache_path.unlink()

        # Create new factory with updated file (different content)
        updated_objects = [
            TestS3Object(
                key="test-bucket/changing_file.txt",
                bytes=updated_content,
            )
        ]
        updated_test_client_factory = make_client_config_and_upload_objects(updated_objects)

        # Download again with updated factory - should re-download due to missing cache
        download_distributed(requests, updated_test_client_factory, testing_info=testing_info)

        # Verify updated content
        for node_id in range(testing_info.num_fake_nodes):
            file_path = resolve_path(dest_path, str(node_id), is_test=True)
            assert file_path.exists()
            assert file_path.read_bytes() == updated_content

    def test_concurrent_download_requests(self, testing_info: SingleNodeTestingInfo):
        """Test behavior with overlapping/concurrent download requests for same files."""

        # Create test objects for concurrent downloads
        test_objects = []
        for i in range(5):
            test_objects.append(
                TestS3Object(
                    key=f"test-bucket/concurrent_{i}.txt",
                    bytes=f"Content {i}".encode(),
                )
            )
        test_client_factory = make_client_config_and_upload_objects(test_objects)

        # Create overlapping requests (some files requested multiple times with different destinations)
        requests = []
        for i in range(5):
            requests.append(
                DownloadRequest(
                    value=ObjectDownloadRequest(
                        uri=f"test-bucket/concurrent_{i}.txt",
                        destination=pathlib.Path(f"/tmp/test_output/dest_a/file_{i}.txt"),
                    )
                )
            )
            # Request same file to different location
            requests.append(
                DownloadRequest(
                    value=ObjectDownloadRequest(
                        uri=f"test-bucket/concurrent_{i}.txt",
                        destination=pathlib.Path(f"/tmp/test_output/dest_b/file_{i}.txt"),
                    )
                )
            )

        download_distributed(requests, test_client_factory, testing_info=testing_info)

        # Verify both destinations have the files
        for node_id in range(testing_info.num_fake_nodes):
            for i in range(5):
                dest_a = resolve_path(pathlib.Path(f"/tmp/test_output/dest_a/file_{i}.txt"), str(node_id), is_test=True)
                dest_b = resolve_path(pathlib.Path(f"/tmp/test_output/dest_b/file_{i}.txt"), str(node_id), is_test=True)

                assert dest_a.exists()
                assert dest_b.exists()
                assert dest_a.read_bytes() == f"Content {i}".encode()
                assert dest_b.read_bytes() == f"Content {i}".encode()

    def test_very_large_chunked_file(self, testing_info: SingleNodeTestingInfo):
        """Test downloading a very large file that creates many chunks."""

        large_content = b"Z" * (10 * 1024 * 1024)
        # Create test objects for concurrent downloads
        test_objects = []
        for _ in range(2):
            test_objects.append(
                TestS3Object(
                    key="test-bucket/very_large.dat",
                    bytes=large_content,
                )
            )
        test_client_factory = make_client_config_and_upload_objects(test_objects)

        # Create a large file that will create many chunks (10MB with 512KB chunks = ~20 chunks)
        requests = [
            DownloadRequest(
                value=ObjectDownloadRequest(
                    uri="test-bucket/very_large.dat",
                    destination=pathlib.Path("/tmp/test_output/very_large.dat"),
                )
            )
        ]

        # Use 512KB chunks to create many chunks
        download_distributed(requests, test_client_factory, chunk_size_bytes=512 * 1024, testing_info=testing_info)

        # Verify file integrity on all nodes
        for node_id in range(testing_info.num_fake_nodes):
            file_path = resolve_path(pathlib.Path("/tmp/test_output/very_large.dat"), str(node_id), is_test=True)
            assert file_path.exists()
            assert file_path.read_bytes() == large_content
            assert len(file_path.read_bytes()) == 10 * 1024 * 1024

    def test_many_small_files(self, testing_info: SingleNodeTestingInfo):
        """Test system performance with many small files (no chunking needed)."""

        num_small_files = 50

        test_objects = []
        # Create many small files
        for i in range(num_small_files):
            content = f"Small file number {i} content".encode()
            test_objects.append(
                TestS3Object(
                    key=f"test-bucket/small_{i:03d}.txt",
                    bytes=content,
                )
            )
        test_client_factory = make_client_config_and_upload_objects(test_objects)

        requests = []
        for i in range(num_small_files):
            requests.append(
                DownloadRequest(
                    value=ObjectDownloadRequest(
                        uri=f"test-bucket/small_{i:03d}.txt",
                        destination=pathlib.Path(f"/tmp/test_output/small_{i:03d}.txt"),
                    )
                )
            )

        download_distributed(requests, test_client_factory, testing_info=testing_info, node_parallelism=10)

        # Verify all files are downloaded on all nodes
        for node_id in range(testing_info.num_fake_nodes):
            for i in range(num_small_files):
                file_path = resolve_path(
                    pathlib.Path(f"/tmp/test_output/small_{i:03d}.txt"), str(node_id), is_test=True
                )
                assert file_path.exists()
                expected_content = f"Small file number {i} content".encode()
                assert file_path.read_bytes() == expected_content
