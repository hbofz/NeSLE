"""Parse the bundled Stable Retro Level 1-1 FCS state and verify known field values.

The reference values come from inspecting the file with a hex dump (see
project_savestate_reset_plan.md). If FCS layout for FCEUX state.cpp changes upstream,
update both this test and the parser together.
"""
from __future__ import annotations

import gzip
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "docs" / "data" / "smb_level1_1.state"


def _load_decompressed_state() -> bytes:
    return gzip.decompress(STATE_PATH.read_bytes())


class FcsParserTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from nesle._cuda_core import parse_fcs_state  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"_cuda_core not built with parse_fcs_state: {exc}")
        self.parse = parse_fcs_state
        self.bytes_in = _load_decompressed_state()

    def test_known_cpu_registers(self) -> None:
        snap = self.parse(self.bytes_in)
        self.assertEqual(snap["pc"], 0x8057)
        self.assertEqual(snap["a"], 0x90)
        self.assertEqual(snap["x"], 0x00)
        self.assertEqual(snap["y"], 0x01)
        self.assertEqual(snap["sp"], 0xFF)
        self.assertEqual(snap["p"], 0xA5)

    def test_known_ppu_registers(self) -> None:
        snap = self.parse(self.bytes_in)
        # PPUR = 90 1e 40 ?? -> ctrl, mask, status
        self.assertEqual(snap["ppu_ctrl"], 0x90)
        self.assertEqual(snap["ppu_mask"], 0x1E)
        self.assertEqual(snap["ppu_status"], 0x40)
        # RADD = 00 08 (LE) -> 0x0800
        self.assertEqual(snap["ppu_v"], 0x0800)
        # TADD = 00 00 (LE) -> 0x0000
        self.assertEqual(snap["ppu_t"], 0x0000)
        # XOFF = 0 (fine x; only low 3 bits kept)
        self.assertEqual(snap["ppu_x"], 0x00)
        # VTGL = 0 (write toggle)
        self.assertEqual(snap["ppu_w"], 0x00)
        # VBUF = 0xff (read buffer)
        self.assertEqual(snap["ppu_read_buffer"], 0xFF)
        # PGEN = 0x90 (open bus)
        self.assertEqual(snap["ppu_open_bus"], 0x90)

    def test_memory_buffer_sizes(self) -> None:
        snap = self.parse(self.bytes_in)
        self.assertEqual(len(snap["cpu_ram"]), 2048)
        self.assertEqual(len(snap["prg_ram"]), 8192)
        self.assertEqual(len(snap["nametable_ram"]), 2048)
        self.assertEqual(len(snap["palette_ram"]), 32)
        self.assertEqual(len(snap["oam"]), 256)

    def test_smb_ram_signals_are_in_gameplay(self) -> None:
        """The saved state was captured at the start of W1-1, so SMB's RAM should reflect
        the gameplay-ready scene (OperMode=1, World=1, Level/Area=1, full time)."""
        snap = self.parse(self.bytes_in)
        ram = snap["cpu_ram"]
        # OperMode at 0x0770 should be 1 (main game) — sidesteps the title-screen bug.
        self.assertEqual(ram[0x0770], 1, "OperMode should be 1 (main game) at this snapshot")
        # World index ($075F) = 0 means World 1; Level ($075C) = 0 means stage 1; Area = 0.
        self.assertEqual(ram[0x075F], 0, "World index should be 0 (World 1)")
        self.assertEqual(ram[0x075C], 0, "Level index should be 0 (stage 1)")
        # Time digits at $07F8-$07FA should sum to a non-zero total.
        time = ram[0x07F8] * 100 + ram[0x07F9] * 10 + ram[0x07FA]
        self.assertGreater(time, 0, "Game timer should be running")

    def test_palette_first_entry_is_background_color(self) -> None:
        """SMB W1-1 uses sky blue 0x22 as the universal background in PRAM[0]."""
        snap = self.parse(self.bytes_in)
        self.assertEqual(snap["palette_ram"][0], 0x22)

    def test_rejects_truncated_input(self) -> None:
        with self.assertRaises(Exception):
            self.parse(self.bytes_in[:32])

    def test_rejects_bad_magic(self) -> None:
        bad = b"XXXX" + self.bytes_in[4:]
        with self.assertRaises(Exception):
            self.parse(bad)


if __name__ == "__main__":
    unittest.main()
