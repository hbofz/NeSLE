import unittest

from nesle.rom import CHR_BANK_SIZE, PRG_BANK_SIZE, NametableArrangement, parse_ines


def make_rom(flags6=0, flags7=0, flags8=0, flags9=0, prg_banks=2, chr_banks=1, trainer=False):
    header = bytearray(b"NES\x1a")
    header.extend([prg_banks, chr_banks, flags6 | (0x04 if trainer else 0), flags7])
    header.extend([flags8, flags9])
    header.extend(b"\x00" * 6)
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
        self.assertEqual(rom.submapper, 0)
        self.assertEqual(rom.prg_rom_banks, 2)
        self.assertEqual(rom.chr_rom_banks, 1)
        self.assertEqual(rom.prg_rom_size, 2 * PRG_BANK_SIZE)
        self.assertEqual(rom.chr_rom_size, CHR_BANK_SIZE)
        self.assertTrue(rom.is_nrom)
        self.assertTrue(rom.is_supported_mario_target)

    def test_nes2_nrom_supported_mario_target(self):
        rom = parse_ines(make_rom(flags7=0x08))
        self.assertTrue(rom.is_nes2)
        self.assertTrue(rom.is_supported_mario_target)

    def test_nes2_submapper_not_supported_mario_target(self):
        rom = parse_ines(make_rom(flags7=0x08, flags8=0x10))
        self.assertEqual(rom.submapper, 1)
        self.assertFalse(rom.is_supported_mario_target)

    def test_nes2_extended_sizes_rejected(self):
        with self.assertRaises(ValueError):
            parse_ines(make_rom(flags7=0x08, flags9=1))

    def test_parse_trainer_offset(self):
        rom = parse_ines(make_rom(trainer=True))
        self.assertTrue(rom.has_trainer)
        self.assertEqual(rom.prg_rom[0], ord("P"))
        self.assertFalse(rom.is_supported_mario_target)

    def test_mirroring_bits(self):
        horizontal = parse_ines(make_rom(flags6=0))
        vertical = parse_ines(make_rom(flags6=1))
        four = parse_ines(make_rom(flags6=8))
        self.assertEqual(horizontal.nametable_arrangement, NametableArrangement.HORIZONTAL)
        self.assertEqual(vertical.nametable_arrangement, NametableArrangement.VERTICAL)
        self.assertEqual(four.nametable_arrangement, NametableArrangement.FOUR_SCREEN)

    def test_invalid_magic(self):
        with self.assertRaises(ValueError):
            parse_ines(b"bad")

    def test_truncated_body(self):
        with self.assertRaises(ValueError):
            parse_ines(make_rom()[:-1])


def mapper_header(n):
    """(flags6, flags7) bytes that declare mapper `n` in a plain iNES header."""
    return ((n & 0x0F) << 4, (n >> 4) << 4)


def make_mapper_rom(n, flags6=0, **kwargs):
    base6, flags7 = mapper_header(n)
    return parse_ines(make_rom(flags6=base6 | flags6, flags7=flags7, **kwargs))


class MapperSupportTests(unittest.TestCase):
    def test_nrom_still_supported(self):
        self.assertTrue(parse_ines(make_rom()).is_supported)
        self.assertTrue(parse_ines(make_rom(prg_banks=1)).is_supported)
        self.assertIsNone(parse_ines(make_rom()).unsupported_reason)

    def test_uxrom_family_supported(self):
        for mapper, name in (
            (2, "UxROM"),
            (11, "Color Dreams (UNROM variant)"),
            (30, "UNROM 512"),
            (94, "UN1ROM"),
        ):
            rom = make_mapper_rom(mapper)
            self.assertEqual(rom.mapper, mapper, name)
            self.assertTrue(rom.is_supported, name)
            self.assertEqual(rom.mapper_name, name)
            self.assertTrue(rom.is_uxrom, name)
            self.assertFalse(rom.is_nrom, name)

    def test_bnrom_and_nina_split_on_chr_presence(self):
        bnrom = make_mapper_rom(34, chr_banks=0)
        self.assertEqual(bnrom.mapper, 34)
        self.assertTrue(bnrom.has_chr_ram)
        self.assertTrue(bnrom.is_supported)

        nina = make_mapper_rom(34, chr_banks=4)
        self.assertEqual(nina.mapper, 34)
        self.assertFalse(nina.has_chr_ram)
        self.assertTrue(nina.is_supported)
        # The device masks CHR ROM, which is only exact for a power of two.
        odd = make_mapper_rom(34, chr_banks=3)
        self.assertFalse(odd.is_supported)
        self.assertIn("iNES 34", odd.unsupported_reason)

    def test_unsupported_mappers(self):
        for mapper in (1, 3, 4, 7, 9, 66, 71):
            rom = make_mapper_rom(mapper)
            self.assertFalse(rom.is_supported, mapper)
            self.assertIn("unsupported mapper", rom.unsupported_reason)
            self.assertIn(str(mapper), rom.unsupported_reason)

    def test_uxrom_prg_size_rules(self):
        # Needs at least 32 KB. iNES PRG sizes are always a whole number of
        # 16 KB banks, so the page-boundary rule can only bite on NES 2.0
        # exponent sizes, which parse_ines rejects earlier.
        self.assertFalse(make_mapper_rom(2, prg_banks=1).is_supported)
        self.assertTrue(make_mapper_rom(2, prg_banks=2).is_supported)
        self.assertTrue(make_mapper_rom(2, prg_banks=8).is_supported)
        # BNROM counts 32 KB pages instead.
        self.assertFalse(make_mapper_rom(34, prg_banks=1, chr_banks=0).is_supported)
        self.assertTrue(make_mapper_rom(34, prg_banks=2, chr_banks=0).is_supported)

    def test_trainer_never_supported(self):
        rom = make_mapper_rom(2, trainer=True)
        self.assertFalse(rom.is_supported)
        self.assertEqual(rom.unsupported_reason, "ROM trainers are not supported")

    def test_unrom512_one_screen_header(self):
        # Header bit 3 with bit 0 clear asks for one-screen mirroring on
        # mapper 30; with bit 0 also set there is no four-screen hardware, so
        # it falls back to the H/V bit.
        single = make_mapper_rom(30, flags6=0x08)
        self.assertEqual(single.mapper, 30)
        self.assertEqual(single.nametable_arrangement, NametableArrangement.SINGLE_SCREEN_LOWER)
        four = make_mapper_rom(30, flags6=0x09)
        self.assertEqual(four.nametable_arrangement, NametableArrangement.VERTICAL)
        # Other UxROM boards ignore the four-screen bit the same way.
        other = make_mapper_rom(2, flags6=0x08)
        self.assertEqual(other.mapper, 2)
        self.assertEqual(other.nametable_arrangement, NametableArrangement.HORIZONTAL)

    def test_mario_target_still_nrom_only(self):
        self.assertFalse(make_mapper_rom(2).is_supported_mario_target)


if __name__ == "__main__":
    unittest.main()
