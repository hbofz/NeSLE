import unittest

import numpy as np

from nesle.native_ppo import NativePPOConfig, compute_gae


try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


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


if __name__ == "__main__":
    unittest.main()
