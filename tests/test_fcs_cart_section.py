"""Cart-section tags: CHRR, LATC and REGS.

These are what make a UxROM/BNROM save resumable at all: CHRR is the 8 KB of
pattern tables, LATC is the bank latch for mappers 2 and 94, REGS[0] is the
32 KB page for mapper 34. The states are built here rather than bundled, so the
test carries no ROM-derived data.
"""
from __future__ import annotations

import struct
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "docs" / "data" / "smb_level1_1.state"


def _sub(name: str, payload: bytes) -> bytes:
    return name.ljust(4, "\0").encode("latin-1") + struct.pack("<I", len(payload)) + payload


def _section(tag: int, payload: bytes) -> bytes:
    return bytes([tag]) + struct.pack("<I", len(payload)) + payload


def build_state(cart: dict[str, bytes]) -> bytes:
    """A minimal but complete FCS\\xff state with the given cart sub-chunks.

    The parser requires CPU and PPU sections with their core tags, which is what
    #23 added, so all of them are present.
    """
    cpu = (
        _sub("PC", struct.pack("<H", 0xC05B))
        + _sub("A", bytes([0x3E]))
        + _sub("S", bytes([0xFF]))
        + _sub("RAM", bytes(2048))
    )
    ppu = (
        _sub("NTAR", bytes(2048))
        + _sub("PRAM", bytes(32))
        + _sub("SPRA", bytes(256))
        + _sub("PPUR", bytes([0xB0, 0x1E, 0x40, 0x00]))
    )
    cart_section = b"".join(_sub(name, payload) for name, payload in cart.items())
    body = _section(1, cpu) + _section(3, ppu)
    if cart_section:
        body += _section(16, cart_section)
    return b"FCS\xff" + bytes(12) + body


class CartSectionTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from nesle._cuda_core import parse_fcs_state  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"_cuda_core not built with parse_fcs_state: {exc}")
        self.parse = parse_fcs_state

    def test_chr_ram_is_parsed(self) -> None:
        pattern = bytes((i * 7) & 0xFF for i in range(8192))
        snap = self.parse(build_state({"CHRR": pattern}))
        self.assertTrue(snap["has_chr_ram"])
        self.assertEqual(bytes(snap["chr_ram"]), pattern)

    def test_latch_is_parsed(self) -> None:
        # What every Contra save on a real cartridge holds.
        snap = self.parse(build_state({"LATC": bytes([0x01])}))
        self.assertTrue(snap["has_cart_latch"])
        self.assertEqual(snap["cart_latch"], 0x01)

    def test_un1rom_latch_is_kept_raw(self) -> None:
        # 0x18 >> 2 == 6 on mapper 94; the parser must not pre-shift it, because
        # the shift belongs to the board, not the file.
        snap = self.parse(build_state({"LATC": bytes([0x18])}))
        self.assertEqual(snap["cart_latch"], 0x18)

    def test_bnrom_regs_first_byte_is_the_page(self) -> None:
        snap = self.parse(build_state({"REGS": bytes([0x03, 0x00, 0x01])}))
        self.assertTrue(snap["has_cart_latch"])
        self.assertEqual(snap["cart_latch"], 0x03)

    def test_state_without_cart_section_has_no_latch(self) -> None:
        # NROM saves carry no bank register; the batch must fall back to bank 0
        # rather than reading an uninitialised byte.
        snap = self.parse(build_state({}))
        self.assertFalse(snap["has_cart_latch"])
        self.assertFalse(snap["has_chr_ram"])

    def test_chr_rom_board_has_no_chr_ram(self) -> None:
        snap = self.parse(build_state({"WRAM": bytes(8192)}))
        self.assertFalse(snap["has_chr_ram"])
        # The buffer is still present, just not claimed by the file.
        self.assertEqual(len(snap["chr_ram"]), 8192)

    def test_wrong_sized_chrr_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            self.parse(build_state({"CHRR": bytes(4096)}))

    def test_odd_sized_latc_is_ignored(self) -> None:
        # A multi-byte LATC is not the tag we know; ignore it rather than throw,
        # so an unfamiliar save still loads.
        snap = self.parse(build_state({"LATC": bytes([1, 2])}))
        self.assertFalse(snap["has_cart_latch"])

    def test_bundled_state_reports_no_cart_section(self) -> None:
        import gzip

        snap = self.parse(gzip.decompress(STATE_PATH.read_bytes()))
        self.assertFalse(snap["has_cart_latch"])
        self.assertFalse(snap["has_chr_ram"])


if __name__ == "__main__":
    unittest.main()
