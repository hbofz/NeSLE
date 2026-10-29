"""Throughput of nes-py / gym-super-mario-bros, for comparison with NeSLE.

This is the widely-used CPU Mario stack (the `legacy-mario` extra). It needs an
OLD environment (gym<0.26, numpy<2) that conflicts with NeSLE's main venv, so
run it in a dedicated venv. numpy 1.26 ships wheels up to Python 3.12; newer
Pythons build it from source, which works on Linux with a C compiler. Windows:

    py -3.10 -m venv .venv-nespy
    .venv-nespy/Scripts/pip install "numpy<2" "gym>=0.25,<0.26" "nes-py>=8.2,<8.3" "gym-super-mario-bros>=7.4,<7.5"
    .venv-nespy/Scripts/python benchmarks/nespy_baseline.py
    .venv-nespy/Scripts/python benchmarks/nespy_baseline.py --workers 12

Linux (the published GTX 1050 Ti figures used Ubuntu's Python 3.14 this way;
nes-py compiles its C++ core at install, and the build moves results ~11%):

    python3 -m venv .venv-nespy
    .venv-nespy/bin/pip install "numpy<2" "gym>=0.25,<0.26" "nes-py>=8.2,<8.3" "gym-super-mario-bros>=7.4,<7.5"
    .venv-nespy/bin/python benchmarks/nespy_baseline.py --workers 12

Reports raw frames/s and the frameskip-4 env-steps/s equivalent used in
NeSLE's GPU-vs-CPU table (one NeSLE env-step = 4 NES frames). Frames include
the ones gym-super-mario-bros fast-forwards inside a single step() to skip
deaths and level transitions: they are emulation work, and NeSLE counts them.

``--workers N`` runs N independent processes, one env each, to load every core.
Each warms up on its own, then all of them time the same wall-clock windows
behind a barrier, so a window's summed rate is real concurrent throughput. The
result is the median window. Separate processes also avoid gym 0.25's
AsyncVectorEnv, whose RNG deepcopy raises
``RandomNumberGenerator._generator_ctor() takes from 0 to 1 positional
arguments`` on numpy >= 1.25.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import time

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

# Long enough for a laptop CPU with every thread loaded to fall from turbo to
# its sustained clock (15-20 s on the i7-8750H), which is what training sees.
WARMUP_SECONDS = 30
WINDOW_SECONDS = 10
WINDOWS = 5
RIGHT = 1  # index of ['right'] in RIGHT_ONLY; NeSLE's benchmarks hold RIGHT too


def step_for(env, seconds: float) -> float:
    """Step for `seconds` of wall time and return emulated frames/s."""
    nes = env.unwrapped
    steps, skipped, start = 0, nes.skipped_frames, time.perf_counter()
    while (elapsed := time.perf_counter() - start) < seconds:
        _, _, done, _ = env.step(RIGHT)
        if done:
            env.reset()
        steps += 1
    return (steps + nes.skipped_frames - skipped) / elapsed


def worker(barrier, results) -> None:
    env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-v0"), RIGHT_ONLY)
    nes, advance = env.unwrapped, env.unwrapped._frame_advance
    nes.skipped_frames = 0

    def counted_advance(action):  # every frame step() emulates besides its own
        nes.skipped_frames += 1
        advance(action)

    nes._frame_advance = counted_advance
    env.reset()
    step_for(env, WARMUP_SECONDS)
    for window in range(WINDOWS):
        barrier.wait()
        results.put((window, step_for(env, WINDOW_SECONDS)))
    env.close()


def main() -> None:
    p = argparse.ArgumentParser(description="nes-py throughput baseline")
    p.add_argument("--workers", type=int, default=1,
                   help="independent processes stepping at once (default 1)")
    workers = p.parse_args().workers
    if workers < 1:
        p.error("--workers must be at least 1")

    barrier, results = mp.Barrier(workers), mp.Queue()
    # daemon: a worker that dies leaves the rest stuck at the barrier, and the
    # get() timeout below then exits the parent, which takes them down with it.
    procs = [mp.Process(target=worker, args=(barrier, results), daemon=True)
             for _ in range(workers)]
    for proc in procs:
        proc.start()
    fps = [0.0] * WINDOWS
    for _ in range(workers * WINDOWS):
        window, rate = results.get(timeout=WARMUP_SECONDS + WINDOW_SECONDS + 120)
        fps[window] += rate
    for proc in procs:
        proc.join()

    median = statistics.median(fps)
    print(f"nes-py SuperMarioBros-v0, {workers} worker(s): {WINDOWS} x "
          f"{WINDOW_SECONDS} s windows after {WARMUP_SECONDS} s warmup")
    print("  per-window env-steps/s: " + ", ".join(f"{f / 4:,.0f}" for f in fps))
    print(f"  median {median:,.0f} frames/s (~{median / 4:,.0f} env-steps/s at frameskip 4)")


if __name__ == "__main__":
    main()
