import tempfile
import unittest
from pathlib import Path

import numpy as np

import nesle
from nesle.actions import SIMPLE_MOVEMENT_MASKS
from nesle.env import NesleEnv, NesleVecEnv
from nesle.rom import CHR_BANK_SIZE, PRG_BANK_SIZE

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


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rom_path = Path(self.tmp.name) / "test.nes"
        self.rom_path.write_bytes(make_rom())

    def tearDown(self):
        self.tmp.cleanup()

    def test_make_vec_reset_step_and_render(self):
        env = nesle.make_vec(
            str(self.rom_path),
            num_envs=2,
            backend="synthetic",
            action_space="simple",
            max_episode_steps=2,
        )
        obs = env.reset()
        self.assertEqual(obs.shape, (2, 240, 256, 3))
        self.assertEqual(obs.dtype, np.uint8)
        self.assertEqual(env.action_space.n, len(SIMPLE_MOVEMENT_MASKS))

        obs, rewards, dones, infos = env.step([1, 4])
        self.assertEqual(obs.shape, (2, 240, 256, 3))
        self.assertEqual(rewards.dtype, np.float32)
        self.assertEqual(dones.shape, (2,))
        self.assertIn("reward_components", infos[0])
        self.assertEqual(env.render().shape, (2, 240, 256, 3))

        _, _, dones, infos = env.step([1, 1])
        self.assertTrue(dones.all())
        self.assertIn("terminal_observation", infos[0])
        self.assertEqual(env.reset_infos[0]["backend"], "synthetic")

    def test_step_async_wait(self):
        env = NesleVecEnv(str(self.rom_path), num_envs=1, backend="synthetic")
        env.reset()
        env.step_async([1])
        obs, rewards, dones, infos = env.step_wait()
        self.assertEqual(obs.shape, (1, 240, 256, 3))
        self.assertEqual(rewards.shape, (1,))
        self.assertEqual(dones.shape, (1,))
        self.assertEqual(len(infos), 1)

    def test_single_env_gymnasium_style_api_without_optional_gym(self):
        env = NesleEnv(str(self.rom_path), backend="synthetic")
        obs, info = env.reset(seed=123)
        self.assertEqual(obs.shape, (240, 256, 3))
        self.assertEqual(info["backend"], "synthetic")
        obs, reward, terminated, truncated, info = env.step(1)
        self.assertEqual(obs.shape, (240, 256, 3))
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("x_pos", info)

    def test_invalid_action_shape_fails(self):
        env = NesleVecEnv(str(self.rom_path), num_envs=2, backend="synthetic")
        env.reset()
        with self.assertRaises(ValueError):
            env.step([1])


if __name__ == "__main__":
    unittest.main()
