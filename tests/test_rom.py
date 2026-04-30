import unittest

from nesle.rom import CHR_BANK_SIZE, PRG_BANK_SIZE, NametableArrangement, parse_ines


def make_rom(flags6=0, flags7=0, prg_banks=2, chr_banks=1, trainer=False):
    header = bytearray(b"NES\x1a")
    header.extend([prg_banks, chr_banks, flags6 | (0x04 if trainer else 0), flags7])
    header.extend(b"\x00" * 8)
    body = bytearray()
    if trainer:
        body.extend(b"T" * 512)
    body.extend(b"P" * (prg_banks * PRG_BANK_SIZE))
    body.extend(b"C" * (chr_banks * CHR_BANK_SIZE))
    return bytes(header + body)


class RomTests(unittest.TestCase):
    def test_parse_nrom_256(self):
        rom = parse_ines(make_rom())
        self.assertEqual(rom.mapper, 0)
        self.assertEqual(rom.prg_rom_banks, 2)
        self.assertEqual(rom.chr_rom_banks, 1)
        self.assertEqual(rom.prg_rom_size, 2 * PRG_BANK_SIZE)
        self.assertEqual(rom.chr_rom_size, CHR_BANK_SIZE)
        self.assertTrue(rom.is_nrom)
        self.assertTrue(rom.is_supported_mario_target)

    def test_nes2_not_supported_mario_target(self):
        rom = parse_ines(make_rom(flags7=0x08))
        self.assertFalse(rom.is_supported_mario_target)

    def test_parse_trainer_offset(self):
        rom = parse_ines(make_rom(trainer=True))
        self.assertTrue(rom.has_trainer)
        self.assertEqual(rom.prg_rom[0], ord("P"))
        self.assertFalse(rom.is_supported_mario_target)

    def test_mirroring_bits(self):
        vertical = parse_ines(make_rom(flags6=0))
        horizontal = parse_ines(make_rom(flags6=1))
        four = parse_ines(make_rom(flags6=8))
        self.assertEqual(vertical.nametable_arrangement, NametableArrangement.VERTICAL)
        self.assertEqual(horizontal.nametable_arrangement, NametableArrangement.HORIZONTAL)
        self.assertEqual(four.nametable_arrangement, NametableArrangement.FOUR_SCREEN)

    def test_invalid_magic(self):
        with self.assertRaises(ValueError):
            parse_ines(b"bad")

    def test_truncated_body(self):
        with self.assertRaises(ValueError):
            parse_ines(make_rom()[:-1])


if __name__ == "__main__":
    unittest.main()
