"""frames_device(): pixel observations as device tensors, no host copy.

render_device() launches the render kernel; frames_device() returns a
CudaDeviceArrayView over the on-device RGB buffer, shape (N, 240, 256, 3)
uint8, consumable by torch via DLPack / __cuda_array_interface__. The device
view must agree byte-for-byte with the host render() path.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ROM_PATH = REPO_ROOT / "Super Mario Bros. (World).nes"
STATE_PATH = REPO_ROOT / "docs" / "data" / "smb_level1_1.state"


def _require_cuda_and_rom() -> None:
    if not ROM_PATH.is_file():
        raise unittest.SkipTest(f"SMB ROM not found at {ROM_PATH}")
    try:
        import nesle._cuda_core  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest(f"_cuda_core not available: {exc}")


class DeviceFramesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_cuda_and_rom()

    def test_frames_device_matches_host_render(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        if not torch.cuda.is_available():
            self.skipTest("torch CUDA not available")
        from nesle._cuda_core import CudaBatch

        batch = CudaBatch(2, 4, ROM_PATH.read_bytes())
        batch.reset()
        right = np.array([0x80, 0x00], dtype=np.uint8)
        for _ in range(20):
            batch.step(right, render_frame=False, copy_obs=False)

        host = batch.render().copy()
        batch.render_device()
        dev = torch.utils.dlpack.from_dlpack(batch.frames_device())

        self.assertEqual(tuple(dev.shape), (2, 240, 256, 3))
        self.assertEqual(dev.dtype, torch.uint8)
        self.assertEqual(dev.device.type, "cuda")
        np.testing.assert_array_equal(dev.cpu().numpy(), host)
        # Both envs rendered something (not an all-zero buffer).
        self.assertGreater(int((dev != 0).sum().item()), 1000)

        # Deterministic teardown: drop the DLPack tensor BEFORE the batch and
        # flush the device. Leaving both to interpreter GC lets the tensor's
        # capsule deleter run after the batch owning the memory is destroyed,
        # which poisons later tests in the same process.
        import gc

        del dev
        gc.collect()
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
