"""Multi-level curriculum: NesleVecEnv with a list of save-state paths.

The 8 packaged states are World N - 1 for N in 1..8. After construction we check that
each env's RAM[0x075F] (World index, 0-based) matches the snapshot it was assigned.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import nesle

REPO_ROOT = Path(__file__).resolve().parents[1]
ROM_PATH = REPO_ROOT / "Super Mario Bros. (World).nes"
STATE_DIR = REPO_ROOT / "docs" / "data"
ALL_LEVELS = [STATE_DIR / f"smb_level{w}_1.state" for w in range(1, 9)]


def _require_cuda_and_assets() -> None:
    if not ROM_PATH.is_file():
        raise unittest.SkipTest(f"SMB ROM not found at {ROM_PATH}")
    for p in ALL_LEVELS:
        if not p.is_file():
            raise unittest.SkipTest(f"missing reset state {p}")
    try:
        import nesle._cuda_core  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest(f"_cuda_core not available: {exc}")


class MultiLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_cuda_and_assets()

    def test_round_robin_assignment_across_8_levels(self) -> None:
        env = nesle.make_vec(
            rom_path=str(ROM_PATH),
            num_envs=16,
            backend="cuda",
            observation_mode="ram",
            reset_state_paths=[str(p) for p in ALL_LEVELS],
        )
        self.assertTrue(env._cuda_batch.has_snapshot)
        self.assertEqual(env._cuda_batch.num_levels, 8)
        obs = env.reset()
        self.assertEqual(obs.shape, (16, 2048))
        worlds = obs[:, 0x075F]  # 0-based; level i+1 -> world index i
        # Round-robin: envs 0..15 should see worlds 0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7
        expected = np.tile(np.arange(8, dtype=np.uint8), 2)
        np.testing.assert_array_equal(worlds, expected)
        env.close()

    def test_explicit_env_to_level_assignment(self) -> None:
        # Assign all 4 envs to World 3 (state index 2)
        env = nesle.make_vec(
            rom_path=str(ROM_PATH),
            num_envs=4,
            backend="cuda",
            observation_mode="ram",
            reset_state_paths=[str(p) for p in ALL_LEVELS],
            env_to_level=[2, 2, 2, 2],
        )
        obs = env.reset()
        np.testing.assert_array_equal(obs[:, 0x075F], [2, 2, 2, 2])
        env.close()

    def test_per_env_reset_preserves_level_assignment(self) -> None:
        """When an env terminates and reset_envs() fires, it should restore to its assigned
        level (not, e.g., level 0)."""
        env = nesle.make_vec(
            rom_path=str(ROM_PATH),
            num_envs=4,
            backend="cuda",
            observation_mode="ram",
            action_space="raw",
            max_episode_steps=2,
            reset_state_paths=[str(p) for p in ALL_LEVELS[:4]],  # worlds 1,2,3,4
        )
        env.reset()
        # Step a few times — max_episode_steps=2 will truncate.
        for _ in range(3):
            obs, _, dones, _ = env.step(np.array([0x00, 0x00, 0x00, 0x00], dtype=np.int64))
            if dones.all():
                break
        self.assertTrue(dones.all())
        # After auto-reset, each env should be back at its OWN level.
        np.testing.assert_array_equal(obs[:, 0x075F], [0, 1, 2, 3])
        env.close()

    def test_mutually_exclusive_single_and_multi_args(self) -> None:
        with self.assertRaises(ValueError):
            nesle.make_vec(
                rom_path=str(ROM_PATH),
                num_envs=2,
                backend="cuda",
                reset_state_path=str(ALL_LEVELS[0]),
                reset_state_paths=[str(p) for p in ALL_LEVELS[:2]],
            )

    def test_env_to_level_requires_multi(self) -> None:
        with self.assertRaises(ValueError):
            nesle.make_vec(
                rom_path=str(ROM_PATH),
                num_envs=2,
                backend="cuda",
                reset_state_path=str(ALL_LEVELS[0]),
                env_to_level=[0, 0],
            )

    def test_single_level_still_works_via_old_arg(self) -> None:
        """Backward-compat: the single-snapshot path uses the original 4-arg CudaBatch ctor."""
        env = nesle.make_vec(
            rom_path=str(ROM_PATH),
            num_envs=2,
            backend="cuda",
            observation_mode="ram",
            reset_state_path=str(ALL_LEVELS[0]),
        )
        self.assertEqual(env._cuda_batch.num_levels, 1)
        obs = env.reset()
        np.testing.assert_array_equal(obs[:, 0x075F], [0, 0])
        env.close()


if __name__ == "__main__":
    unittest.main()
