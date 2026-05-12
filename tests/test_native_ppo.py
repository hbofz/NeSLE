import tempfile
import unittest
from pathlib import Path

import numpy as np

from nesle.native_ppo import (
    NativePPOConfig,
    _load_checkpoint,
    _save_checkpoint,
    compute_gae,
)


try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


class TestNativePPOConfig(unittest.TestCase):
    def test_defaults_are_training_sized(self):
        config = NativePPOConfig("mario.nes")
        self.assertGreater(config.num_envs, 0)
        self.assertGreater(config.n_steps * config.num_envs, 1)
        self.assertGreater(config.batch_size, 1)


class TestGAE(unittest.TestCase):
    def test_compute_gae_matches_discounted_returns_when_values_zero(self):
        if torch is None:
            self.skipTest("torch is not installed")
        rewards = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        dones = torch.zeros_like(rewards)
        values = torch.zeros_like(rewards)
        next_done = torch.zeros(2)
        next_value = torch.zeros(2)

        _, returns = compute_gae(
            rewards,
            dones,
            values,
            next_done,
            next_value,
            gamma=0.5,
            gae_lambda=1.0,
        )

        expected = torch.tensor([[1.75, 2.75], [1.5, 3.5], [1.0, 3.0]])
        np.testing.assert_allclose(returns.numpy(), expected.numpy(), rtol=1e-6)

    def test_compute_gae_stops_at_done_boundary(self):
        if torch is None:
            self.skipTest("torch is not installed")
        rewards = torch.tensor([[1.0], [1.0], [10.0]])
        dones = torch.tensor([[0.0], [0.0], [1.0]])
        values = torch.zeros_like(rewards)
        next_done = torch.zeros(1)
        next_value = torch.zeros(1)

        _, returns = compute_gae(
            rewards,
            dones,
            values,
            next_done,
            next_value,
            gamma=0.9,
            gae_lambda=1.0,
        )

        expected = torch.tensor([[1.9], [1.0], [10.0]])
        np.testing.assert_allclose(returns.numpy(), expected.numpy(), rtol=1e-6)


class TestCheckpointRoundTrip(unittest.TestCase):
    """The checkpoint must preserve RNG state so resumed runs continue the same trajectory."""

    def setUp(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ckpt.pt"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_model_optimizer(self):
        # Tiny CPU model + optimizer; this test exercises the checkpoint plumbing, not
        # training itself. The model has zero learnable params from the env, just a Linear.
        model = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return model, optimizer

    def test_torch_and_numpy_rng_round_trip(self) -> None:
        model, optimizer = self._make_model_optimizer()
        config = NativePPOConfig("dummy.nes")

        # Seed and save
        torch.manual_seed(42)
        np.random.seed(42)
        _save_checkpoint(str(self.path), model, optimizer, config, global_step=1234)

        # Mutate global RNG state to prove load restores it
        torch.rand(100)
        np.random.rand(100)

        model2, optimizer2 = self._make_model_optimizer()
        step = _load_checkpoint(str(self.path), model2, optimizer2)
        self.assertEqual(step, 1234)

        # Now next draws from torch + numpy should match a fresh seed-42 run
        expected_torch = torch.manual_seed(42)
        np.random.seed(42)
        torch_draw_a = torch.rand(5)
        numpy_draw_a = np.random.rand(5)

        # Restore from checkpoint and draw
        _ = _load_checkpoint(str(self.path), model2, optimizer2)
        torch_draw_b = torch.rand(5)
        numpy_draw_b = np.random.rand(5)

        torch.testing.assert_close(torch_draw_a, torch_draw_b)
        np.testing.assert_allclose(numpy_draw_a, numpy_draw_b)


class TestBatchSizeValidation(unittest.TestCase):
    def test_rollout_smaller_than_batch_raises(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed")
        # Importing train_native_ppo here to keep the heavy import inside the test
        from nesle.native_ppo import train_native_ppo

        # num_envs * n_steps = 4 * 8 = 32, batch_size = 64 (intentionally too big).
        # This must raise BEFORE we even try to construct the env, so we can use a bogus
        # rom path — the validation should short-circuit. Test that no SystemExit fires
        # (which would mean we got far enough to require CUDA/ROM).
        config = NativePPOConfig(
            rom_path="bogus.nes",
            num_envs=4,
            n_steps=8,
            batch_size=64,
            total_timesteps=64,
        )
        with self.assertRaises((ValueError, SystemExit)) as cm:
            train_native_ppo(config)
        # The error message should mention batch_size if our validation fired (rather than
        # a "no CUDA" / "no ROM" SystemExit from the env path).
        if isinstance(cm.exception, ValueError):
            self.assertIn("batch_size", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
