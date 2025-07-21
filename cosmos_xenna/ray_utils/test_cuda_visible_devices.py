import types
import pytest

from cosmos_xenna.ray_utils import resources as res


def _inject_nvml_stub(monkeypatch, num_gpus: int = 2):
    """Patch `cosmos_xenna.ray_utils.resources` so that it believes NVML is available
    and pretends that there are `num_gpus` GPUs present. Each GPU is given a predictable
    UUID (``GPU-<index>``) so we can reference them via ``CUDA_VISIBLE_DEVICES``.
    """

    def nvml_init():
        return None

    def nvml_shutdown():
        return None

    def nvml_device_get_count():
        return num_gpus

    def nvml_device_get_handle_by_index(index):
        # The handle can be the index itself for our stubbed purposes.
        return index

    def nvml_device_get_name(handle):
        # Return bytes to match the real NVML interface.
        return b"Fake GPU"

    def nvml_device_get_uuid(handle):
        # Real NVML returns bytes; we can safely return str here because the prod code
        # immediately casts to ``str`` anyway.
        return f"GPU-{handle}"

    pynvml_stub = types.SimpleNamespace(
        nvmlInit=nvml_init,
        nvmlShutdown=nvml_shutdown,
        nvmlDeviceGetCount=nvml_device_get_count,
        nvmlDeviceGetHandleByIndex=nvml_device_get_handle_by_index,
        nvmlDeviceGetName=nvml_device_get_name,
        nvmlDeviceGetUUID=nvml_device_get_uuid,
        # The prod code catches ``pynvml.NVMLError`` – provide a stand-in.
        NVMLError=Exception,
    )

    # Ensure the module believes NVML is available and uses our stub.
    monkeypatch.setattr(res, "HAS_NVML", True, raising=True)
    monkeypatch.setattr(res, "pynvml", pynvml_stub, raising=True)


@pytest.fixture()
def patched_resources(monkeypatch):
    """Provide the *resources* module patched with a stubbed NVML implementation."""
    _inject_nvml_stub(monkeypatch)
    return res


def test_cuda_visible_devices_index(monkeypatch, patched_resources):
    """If CUDA_VISIBLE_DEVICES is set to an integer index, only that GPU is returned."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    gpus = patched_resources._get_local_gpu_info()

    assert len(gpus) == 1
    assert gpus[0].index == 1
    assert gpus[0].uuid == "GPU-1"


def test_cuda_visible_devices_uuid(monkeypatch, patched_resources):
    """If CUDA_VISIBLE_DEVICES is set to a GPU UUID, only that GPU is returned."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0")

    gpus = patched_resources._get_local_gpu_info()

    assert len(gpus) == 1
    assert gpus[0].index == 0
    assert gpus[0].uuid == "GPU-0"


def test_cuda_visible_devices_invalid(monkeypatch, patched_resources):
    """An invalid CUDA_VISIBLE_DEVICES entry should raise a ValueError."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "invalid-value")

    with pytest.raises(ValueError):
        _ = patched_resources._get_local_gpu_info()
