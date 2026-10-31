import sys

sys.path.insert(0, "src")


def _hide_cuda_core_without_a_gpu() -> None:
    """Tests skip on ImportError, so report a built extension with no visible GPU as absent."""
    try:
        import nesle._cuda_core as cuda_core
    except ImportError:
        return
    try:
        cuda_core.CudaBatch(1, 1)
    except RuntimeError as exc:
        if "no CUDA-capable device" not in str(exc):
            raise
        del sys.modules["nesle"]._cuda_core
        sys.modules["nesle._cuda_core"] = None


_hide_cuda_core_without_a_gpu()
