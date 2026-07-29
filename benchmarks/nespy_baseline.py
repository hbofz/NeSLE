"""Throughput of nes-py / gym-super-mario-bros, for comparison with NeSLE.

This is the widely-used CPU Mario stack (the `legacy-mario` extra). It needs an
OLD environment — gym<0.26, numpy<2, Python <= 3.11 — which conflicts with
NeSLE's main venv, so run it in a dedicated venv:

    py -3.10 -m venv .venv-nespy
    .venv-nespy/Scripts/pip install "numpy<2" "gym>=0.25,<0.26" "nes-py>=8.2,<8.3" "gym-super-mario-bros>=7.4,<7.5"
    .venv-nespy/Scripts/python benchmarks/nespy_baseline.py

Reports raw frames/s and the frameskip-4 env-steps/s equivalent used in
NeSLE's GPU-vs-CPU table (one NeSLE env-step = 4 NES frames).
"""

from __future__ import annotations

import time

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

WARMUP_FRAMES = 120
TIMED_FRAMES = 2000
RIGHT_B = 3  # index of ['right', 'B'] in RIGHT_ONLY


def main() -> None:
    env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-v0"), RIGHT_ONLY)
    env.reset()
    for _ in range(WARMUP_FRAMES):
        _, _, done, _ = env.step(RIGHT_B)
        if done:
            env.reset()

    start = time.perf_counter()
    for _ in range(TIMED_FRAMES):
        _, _, done, _ = env.step(RIGHT_B)
        if done:
            env.reset()
    elapsed = time.perf_counter() - start
    env.close()

    fps = TIMED_FRAMES / elapsed
    print(f"nes-py SuperMarioBros-v0: {fps:,.0f} frames/s "
          f"(~{fps / 4:,.0f} env-steps/s at frameskip 4)")


if __name__ == "__main__":
    main()
