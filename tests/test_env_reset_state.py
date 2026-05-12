"""End-to-end tests for NesleVecEnv with reset_state_path (the Path B unblock).

These tests require the SMB ROM at the repo root and a CUDA-capable GPU. They are skipped
when either is missing.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import nesle

REPO_ROOT = Path(__file__).resolve().parents[1]
ROM_PATH = REPO_ROOT / "Super Mario Bros. (World).nes"
STATE_PATH = REPO_ROOT / "docs" / "data" / "smb_level1_1.state"


def _require_cuda_and_rom() -> None:
    if not ROM_PATH.is_file():
        raise unittest.SkipTest(f"SMB ROM not found at {ROM_PATH}")
    if not STATE_PATH.is_file():
        raise unittest.SkipTest(f"reset state not found at {STATE_PATH}")
    try:
        import nesle._cuda_core  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest(f"_cuda_core extension not available: {exc}")


class ResetStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_cuda_and_rom()

    def test_initial_reset_lands_in_gameplay(self) -> None:
        env = nesle.make_vec(
            rom_path=str(ROM_PATH),
            num_envs=1,
            backend="cuda",
            observation_mode="ram",
            reset_state_path=str(STATE_PATH),
        )
        obs = env.reset()
        self.assertEqual(obs.shape, (1, 2048))
        ram = obs[0]
        # OperMode=1 means we're in main gameplay; cold-boot would leave this at 0.
        self.assertEqual(ram[0x0770], 1, "OperMode should be 1 (main game)")
        # World 1 / Stage 1 / Area 1 — the bundled snapshot is W1-1.
        self.assertEqual(int(ram[0x075F]), 0, "World index should be 0 (World 1)")
        self.assertEqual(int(ram[0x075C]), 0, "Stage index should be 0 (stage 1)")
        env.close()

    def test_mario_advances_on_right_after_snapshot_reset(self) -> None:
        env = nesle.make_vec(
            rom_path=str(ROM_PATH),
            num_envs=1,
            backend="cuda",
            observation_mode="ram",
            action_space="raw",
            reset_state_path=str(STATE_PATH),
        )
        env.reset()
        ram_initial = env._cuda_batch.ram()[0]
        x0 = int(ram_initial[0x006D]) * 0x100 + int(ram_initial[0x0086])
        # Hold RIGHT (raw mask 0x80) for 30 steps.
        for _ in range(30):
            env.step(np.array([0x80], dtype=np.int64))
        ram_final = env._cuda_batch.ram()[0]
        x1 = int(ram_final[0x006D]) * 0x100 + int(ram_final[0x0086])
        self.assertGreater(x1, x0 + 50, f"Mario should advance >50 px under RIGHT (got x0={x0}, x1={x1})")
        env.close()

    def test_auto_reset_returns_to_snapshot_state(self) -> None:
        """When an env's episode terminates, the auto-reset path should land back in
        W1-1 with the same OperMode=1, not the broken cold-boot title screen."""
        env = nesle.make_vec(
            rom_path=str(ROM_PATH),
            num_envs=2,
            backend="cuda",
            observation_mode="ram",
            action_space="raw",
            max_episode_steps=4,  # force quick truncation via the wrapper
            reset_state_path=str(STATE_PATH),
        )
        env.reset()
        # Step until truncation kicks in.
        for _ in range(5):
            obs, rewards, dones, infos = env.step(np.array([0x00, 0x00], dtype=np.int64))
            if dones.any():
                break
        self.assertTrue(dones.all(), "expected both envs to truncate at max_episode_steps")
        # After auto-reset the returned obs should be post-snapshot-restore for done envs.
        for env_idx in range(2):
            ram = obs[env_idx]
            self.assertEqual(ram[0x0770], 1, f"env {env_idx} should be back in gameplay post-reset")
        env.close()

    def test_reset_state_requires_cuda_backend(self) -> None:
        with self.assertRaises(ValueError):
            nesle.make_vec(
                rom_path=str(ROM_PATH),
                num_envs=1,
                backend="synthetic",
                reset_state_path=str(STATE_PATH),
            )

    def test_has_snapshot_property_exposed(self) -> None:
        env = nesle.make_vec(
            rom_path=str(ROM_PATH),
            num_envs=1,
            backend="cuda",
            observation_mode="ram",
            reset_state_path=str(STATE_PATH),
        )
        self.assertTrue(env._cuda_batch.has_snapshot)
        env.close()

    def test_device_view_outlives_intermediate_batch_reference(self) -> None:
        """Regression for the CudaDeviceArrayView lifetime contract.

        ram_device() / step_device() return views holding bare device pointers owned by
        the CudaBatch. Pybind `keep_alive<0, 1>` should keep the parent CudaBatch alive
        as long as a Python caller still holds the view, even if the original Python
        reference to the batch is dropped. Without keep_alive, dereferencing the view
        after the batch's destructor freed device_ram_ would be a use-after-free.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                self.skipTest("CUDA-enabled torch not available")
        except ImportError:
            self.skipTest("torch is not installed")
        from nesle._cuda_core import CudaBatch
        import gc

        rom_bytes = ROM_PATH.read_bytes()
        snapshot_bytes = __import__("gzip").decompress(STATE_PATH.read_bytes())

        batch = CudaBatch(2, 4, rom_bytes, snapshot_bytes)
        batch.reset()
        # Acquire a torch tensor via DLPack. The view (and now the tensor) should keep
        # the batch alive even after we drop our local reference.
        ram_tensor = torch.utils.dlpack.from_dlpack(batch.ram_device())
        # Drop the only Python reference to the batch.
        del batch
        gc.collect()
        # If keep_alive is working, ram_tensor's device memory is still valid and we
        # can read from it without crashing. Numerical content doesn't matter for this
        # test — we just need a syntactically valid GPU read that would UAF without
        # keep_alive.
        copied = ram_tensor.detach().clone().cpu()
        self.assertEqual(copied.shape, (2, 2048))


if __name__ == "__main__":
    unittest.main()
