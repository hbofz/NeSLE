"""Cold boot through SMB's title screen into gameplay, with no reset state.

The title menu ORs controller 2 into controller 1 before comparing against
Start. The CUDA core used to read every P2 button as held, so the game never
left the title screen (issue #7). Requires the SMB ROM at the repo root.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

ROM_PATH = Path(__file__).resolve().parents[1] / "Super Mario Bros. (World).nes"
START = 0x08
RIGHT = 0x80
OPER_MODE = 0x0770
SAVED_JOYPAD2 = 0x06FD
PLAYER_X = (0x006D, 0x0086)


def _inputs(frame: int) -> int:
    # Idle on the title screen, tap Start once, then walk right.
    if 120 <= frame < 125:
        return START
    return RIGHT if frame >= 300 else 0


def _x(ram) -> int:
    return int(ram[PLAYER_X[0]]) * 256 + int(ram[PLAYER_X[1]])


class TitleScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not ROM_PATH.is_file():
            raise unittest.SkipTest(f"SMB ROM not found at {ROM_PATH}")
        cls.rom = ROM_PATH.read_bytes()

    def _assert_reaches_gameplay(self, step, ram) -> None:
        for frame in range(400):
            step(_inputs(frame))
            self.assertEqual(ram()[SAVED_JOYPAD2], 0, f"phantom P2 input at frame {frame}")
        self.assertEqual(ram()[OPER_MODE], 1, "Start did not leave the title screen")
        self.assertGreater(_x(ram()), 60, "Mario did not walk right")

    def test_host_core(self) -> None:
        try:
            from nesle import _core
        except ImportError as exc:
            self.skipTest(f"_core extension not available: {exc}")
        console = _core.NativeConsole(self.rom)
        self._assert_reaches_gameplay(lambda b: console.step(b, 1, 100000), console.ram)

    def test_cuda_batch(self) -> None:
        try:
            from nesle._cuda_core import CudaBatch
        except ImportError as exc:
            self.skipTest(f"_cuda_core extension not available: {exc}")
        batch = CudaBatch(2, 1, self.rom)
        batch.reset()

        def step(buttons: int) -> None:
            batch.step(np.full(2, buttons, dtype=np.uint8), False, False)

        def ram():
            rows = np.asarray(batch.ram()).reshape(2, -1)
            self.assertTrue(np.array_equal(rows[0], rows[1]), "identical envs diverged")
            return rows[0]

        self._assert_reaches_gameplay(step, ram)


if __name__ == "__main__":
    unittest.main()
