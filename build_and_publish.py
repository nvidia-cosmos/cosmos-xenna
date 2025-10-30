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

"""
Build and publish maturin wheels for multiple architectures.

This script builds wheels for x86_64 and aarch64 Linux targets, then uploads
them to the configured PyPI repository using twine.

Usage:

export TWINE_USERNAME=your_username
export TWINE_PASSWORD=your_password
export TWINE_REPOSITORY_URL=https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi-local

python build_and_publish.py

# Build only (skip publishing)
python build_and_publish.py --build-only

# Publish only (skip building)
python build_and_publish.py --publish-only

# Verbose output
python build_and_publish.py --verbose
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> None:
    """Run a command and exit on failure."""
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"{'=' * 80}")
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(1)
    print(f"\n✅ {description} completed successfully")


def clear_wheels_directory(wheels_dir: Path) -> None:
    """Clear the wheels directory before building."""
    if wheels_dir.exists():
        print(f"\n🧹 Clearing wheels directory: {wheels_dir}")
        shutil.rmtree(wheels_dir)
        print("✅ Wheels directory cleared")
    wheels_dir.mkdir(parents=True, exist_ok=True)


def build_wheels(targets: list[str], use_zig: bool = True) -> None:
    """Build wheels for the specified targets."""
    for target in targets:
        cmd = ["uv", "run", "maturin", "build", "--release", "--target", target]
        if use_zig:
            cmd.append("--zig")

        description = f"Building wheel for {target}"
        run_command(cmd, description)


def publish_wheels(wheels_dir: Path | None = None, verbose: bool = False) -> None:
    """Publish all built wheels using twine.

    Expects the following environment variables to be set:
    - TWINE_USERNAME: PyPI username
    - TWINE_PASSWORD: PyPI password/token
    - TWINE_REPOSITORY_URL: PyPI repository URL
    """
    if wheels_dir is None:
        wheels_dir = Path("target/wheels")

    if not wheels_dir.exists():
        print(f"\n❌ Error: Wheels directory '{wheels_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    wheel_files = list(wheels_dir.glob("*.whl"))
    if not wheel_files:
        print(f"\n❌ Error: No wheel files found in '{wheels_dir}'", file=sys.stderr)
        sys.exit(1)

    print(f"\n📦 Found {len(wheel_files)} wheel(s) to upload:")
    for wheel in wheel_files:
        print(f"  - {wheel.name}")

    # Check that required environment variables are set
    required_vars = ["TWINE_USERNAME", "TWINE_PASSWORD", "TWINE_REPOSITORY_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"\n❌ Error: Missing required environment variables: {', '.join(missing_vars)}", file=sys.stderr)
        print("\nPlease set:", file=sys.stderr)
        for var in missing_vars:
            print(f"  export {var}=<value>", file=sys.stderr)
        sys.exit(1)

    cmd = [
        "uv",
        "run",
        "twine",
        "upload",
        "--non-interactive",
    ]
    if verbose:
        cmd.append("--verbose")
    cmd.extend([str(w) for w in wheel_files])

    description = "Publishing wheels to PyPI repository"
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"{'=' * 80}")
    print(f"Repository: {os.getenv('TWINE_REPOSITORY_URL')}")
    print(f"Username: {os.getenv('TWINE_USERNAME')}")
    print()

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(1)
    print(f"\n✅ {description} completed successfully")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and publish maturin wheels for multiple architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build and publish (credentials from environment)
  export TWINE_USERNAME=myuser
  export TWINE_PASSWORD=mytoken
  export TWINE_REPOSITORY_URL=https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi-local
  python build_and_publish.py

  # Build only (skip publishing)
  python build_and_publish.py --build-only

  # Publish only (skip building)
  python build_and_publish.py --publish-only
        """,
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        default=["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"],
        help="Target architectures to build for (default: x86_64 and aarch64 Linux)",
    )

    parser.add_argument(
        "--wheels-dir",
        type=Path,
        default=Path("target/wheels"),
        help="Directory containing built wheels (default: target/wheels)",
    )

    parser.add_argument(
        "--no-zig",
        action="store_true",
        help="Don't use zig for cross-compilation",
    )

    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build wheels, don't publish",
    )

    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="Only publish existing wheels, don't build",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for twine",
    )

    args = parser.parse_args()

    # Build wheels
    if not args.publish_only:
        clear_wheels_directory(args.wheels_dir)
        print("\n🔨 Building wheels...")
        build_wheels(args.targets, use_zig=not args.no_zig)

    # Publish wheels
    if not args.build_only:
        print("\n📤 Publishing wheels...")
        publish_wheels(wheels_dir=args.wheels_dir, verbose=args.verbose)

    print("\n" + "=" * 80)
    print("🎉 All operations completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
