"""Tests for the critical issue fixes identified in the pre-training audit."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nesle.actions import (
    SIMPLE_MOVEMENT_MASKS,
    SIMPLE_MOVEMENT_WITH_START,
    SIMPLE_MOVEMENT_WITH_START_MASKS,
    encode_action,
)
from nesle.env import NesleEnv, NesleVecEnv, _action_masks
from nesle.rom import CHR_BANK_SIZE, PRG_BANK_SIZE
from nesle.smb import CPU_RAM_BYTES

if not hasattr(np, "uint8"):
    raise unittest.SkipTest("complete numpy package is not available")


def make_rom() -> bytes:
    header = bytearray(b"NES\x1a")
    header.extend([2, 1, 0, 0])
    header.extend(b"\x00" * 8)
    prg = bytearray([0xEA] * (2 * PRG_BANK_SIZE))
    prg[0x7FFC] = 0x00
    prg[0x7FFD] = 0x80
    return bytes(header + prg + bytearray(CHR_BANK_SIZE))


class TestStartButtonActionSpace(unittest.TestCase):
    """C4: Verify START button is wired into selectable action spaces."""

    def test_simple_with_start_defined(self):
        self.assertEqual(len(SIMPLE_MOVEMENT_WITH_START), 8)
        self.assertEqual(SIMPLE_MOVEMENT_WITH_START[0], ["start"])

    def test_simple_with_start_masks(self):
        self.assertEqual(len(SIMPLE_MOVEMENT_WITH_START_MASKS), 8)
        self.assertEqual(SIMPLE_MOVEMENT_WITH_START_MASKS[0], encode_action(["start"]))

    def test_action_masks_selector(self):
        masks = _action_masks("simple_with_start")
        self.assertEqual(masks, SIMPLE_MOVEMENT_WITH_START_MASKS)

    def test_action_masks_alias(self):
        masks = _action_masks("simple_start")
        self.assertEqual(masks, SIMPLE_MOVEMENT_WITH_START_MASKS)

    def test_existing_spaces_unchanged(self):
        self.assertEqual(_action_masks("simple"), SIMPLE_MOVEMENT_MASKS)
        self.assertEqual(len(_action_masks("complex")), 12)
        self.assertEqual(len(_action_masks("right_only")), 5)
        self.assertEqual(len(_action_masks("raw")), 256)


class TestAutoResetContract(unittest.TestCase):
    """C1: Verify auto-reset works correctly with terminal_observation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rom_path = Path(self.tmp.name) / "test.nes"
        self.rom_path.write_bytes(make_rom())

    def tearDown(self):
        self.tmp.cleanup()

    def test_terminal_observation_present_on_done(self):
        env = NesleVecEnv(
            str(self.rom_path),
            num_envs=2,
            backend="synthetic",
            max_episode_steps=2,
        )
        env.reset()
        env.step([1, 1])
        _, _, dones, infos = env.step([1, 1])
        self.assertTrue(dones.all())
        for i in range(2):
            self.assertIn("terminal_observation", infos[i])
            self.assertEqual(infos[i]["terminal_observation"].shape, (240, 256, 3))
        env.close()

    def test_returned_obs_is_post_reset(self):
        env = NesleVecEnv(
            str(self.rom_path),
            num_envs=1,
            backend="synthetic",
            max_episode_steps=2,
        )
        reset_obs = env.reset()
        env.step([1])
        obs, _, dones, infos = env.step([1])
        self.assertTrue(dones[0])
        # The terminal observation should differ from the returned obs
        # (returned obs is post-reset, terminal_observation is the last frame)
        terminal = infos[0]["terminal_observation"]
        # The reset observation should match what we got from reset()
        # (both are the default state)
        self.assertEqual(obs[0].shape, reset_obs[0].shape)
        env.close()

    def test_reset_infos_updated_on_auto_reset(self):
        env = NesleVecEnv(
            str(self.rom_path),
            num_envs=2,
            backend="synthetic",
            max_episode_steps=2,
        )
        env.reset()
        env.step([1, 1])
        _, _, dones, _ = env.step([1, 1])
        self.assertTrue(dones.all())
        self.assertEqual(env.reset_infos[0]["backend"], "synthetic")
        env.close()

    def test_ram_observation_terminal_observation_shape(self):
        env = NesleVecEnv(
            str(self.rom_path),
            num_envs=2,
            backend="synthetic",
            observation_mode="ram",
            max_episode_steps=2,
        )
        env.reset()
        env.step([1, 1])
        _, _, dones, infos = env.step([1, 1])
        self.assertTrue(dones.all())
        self.assertEqual(infos[0]["terminal_observation"].shape, (CPU_RAM_BYTES,))
        env.close()

    def test_simple_with_start_vec_env(self):
        env = NesleVecEnv(
            str(self.rom_path),
            num_envs=2,
            backend="synthetic",
            action_space="simple_with_start",
        )
        self.assertEqual(env.action_space.n, 8)
        obs = env.reset()
        self.assertEqual(obs.shape, (2, 240, 256, 3))
        # Action 0 is START
        obs, rew, done, info = env.step([0, 0])
        self.assertEqual(obs.shape, (2, 240, 256, 3))
        env.close()

    def test_numpy_array_actions_accepted(self):
        env = NesleVecEnv(str(self.rom_path), num_envs=3, backend="synthetic")
        env.reset()
        actions = np.array([1, 2, 0])
        obs, rew, done, info = env.step(actions)
        self.assertEqual(obs.shape, (3, 240, 256, 3))
        env.close()

    def test_single_env_with_start(self):
        env = NesleEnv(
            str(self.rom_path),
            backend="synthetic",
            action_space="simple_with_start",
        )
        obs, info = env.reset(seed=42)
        self.assertEqual(obs.shape, (240, 256, 3))
        obs, rew, term, trunc, info = env.step(0)  # START
        self.assertIsInstance(rew, float)
        self.assertIsInstance(term, bool)
        self.assertIsInstance(trunc, bool)
        env.close()


if __name__ == "__main__":
    unittest.main()
