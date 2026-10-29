"""Regression tests for the render path.

Render freeze: prior to the fix, CudaBatch.render() was a const memcpy of device_frames_; it didn't
launch the render kernel, so calling it after step(render_frame=False) returned a stale
frame from whenever the kernel last ran. NesleVecEnv.render() inherited the bug.

This test asserts the natural contract: stepping the env and then calling render()
returns a frame that reflects the new state, regardless of the throughput flags on step().
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


class RenderFreshnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_cuda_and_rom()

    def test_render_reflects_state_after_throughput_step(self) -> None:
        """step(render_frame=False) followed by render() must still return a fresh frame."""
        if not STATE_PATH.is_file():
            self.skipTest(f"reset state not found at {STATE_PATH}")
        from nesle._cuda_core import CudaBatch
        import gzip

        snapshot = gzip.decompress(STATE_PATH.read_bytes())
        batch = CudaBatch(1, 4, ROM_PATH.read_bytes(), snapshot)
        batch.reset()
        frame_initial = batch.render()[0].copy()
        right = np.array([0x80], dtype=np.uint8)
        for _ in range(30):
            batch.step(right, render_frame=False, copy_obs=False)
        frame_after = batch.render()[0].copy()
        # Mario advances in W1-1 → at least the HUD (TIME counter) and Mario sprite area
        # should differ between frames. Threshold is generous to allow for kernel variance.
        changed = int(np.sum(frame_initial != frame_after))
        self.assertGreater(
            changed, 100,
            f"render() returned a stale frame ({changed} bytes changed; expected >100). "
            "Likely the render kernel wasn't launched before the memcpy.",
        )

    def test_render_matches_inline_render_frame_path(self) -> None:
        """The explicit render() path and step(render_frame=True) path should agree."""
        from nesle._cuda_core import CudaBatch

        rom_bytes = ROM_PATH.read_bytes()
        right = np.array([0x80], dtype=np.uint8)

        # Path A: step with render_frame=False, then render() explicitly.
        a = CudaBatch(1, 4, rom_bytes)
        a.reset()
        for _ in range(20):
            a.step(right, render_frame=False, copy_obs=False)
        frame_a = a.render()[0].copy()

        # Path B: step with render_frame=True, never call render() explicitly.
        b = CudaBatch(1, 4, rom_bytes)
        b.reset()
        for _ in range(20):
            b.step(right, render_frame=True, copy_obs=True)
        frame_b = b.render()[0].copy()

        self.assertTrue(
            np.array_equal(frame_a, frame_b),
            "render() and step(render_frame=True) produced different frames",
        )

    def test_status_bar_survives_scrolling_into_second_nametable(self) -> None:
        """Issue #6: once W1-1 scrolls into the second nametable, frames that carry a
        VRAM update drew the status bar from the playfield's nametable (blank sky)."""
        if not STATE_PATH.is_file():
            self.skipTest(f"reset state not found at {STATE_PATH}")
        from nesle._cuda_core import CudaBatch
        import gzip

        batch = CudaBatch(1, 1, ROM_PATH.read_bytes(), gzip.decompress(STATE_PATH.read_bytes()))
        batch.reset()
        glyphs = None
        missing = []
        for frame in range(444):  # Mario is alive for 444 frames under this input script
            # RIGHT+B held, A (jump) for 20 of every 60 frames
            action = np.array([0x82 | (0x01 if frame % 60 < 20 else 0)], dtype=np.uint8)
            batch.step(action, render_frame=False, copy_obs=False)
            # Rows 8-24 hold only the fixed white MARIO / WORLD / TIME labels.
            text = (batch.render()[0, 8:24] > 200).all(axis=2)
            if glyphs is None:
                glyphs = text
                self.assertGreater(int(glyphs.sum()), 300, "no status bar on the first frame")
            elif text[glyphs].mean() <= 0.9:
                missing.append(frame)
        self.assertEqual(missing, [], f"status bar missing on {len(missing)} of 444 frames")


if __name__ == "__main__":
    unittest.main()
