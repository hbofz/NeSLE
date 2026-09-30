from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


HEADER_SIZE = 16
TRAINER_SIZE = 512
PRG_BANK_SIZE = 16 * 1024
CHR_BANK_SIZE = 8 * 1024


class NametableArrangement(str, Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    FOUR_SCREEN = "four_screen"
    # UNROM 512 (iNES mapper 30) reuses the four-screen header bit to ask for
    # one-screen mirroring; the CIRAM page is picked at run time by bit 7 of
    # the bank register and is the lower page at power-on.
    SINGLE_SCREEN_LOWER = "single_screen_lower"
    SINGLE_SCREEN_UPPER = "single_screen_upper"


# iNES mapper numbers the batched emulator drives. Mapper 0 has no register at
# all; the rest reduce to a switchable PRG window plus a fixed one.
MAPPER_NROM = 0
MAPPER_UXROM = 2
MAPPER_COLOR_DREAMS = 11
MAPPER_UNROM_512 = 30
MAPPER_BNROM = 34
MAPPER_UN1ROM = 94
MAPPER_UXROM_180 = 180

SUPPORTED_MAPPERS = (
    MAPPER_NROM,
    MAPPER_UXROM,
    MAPPER_COLOR_DREAMS,
    MAPPER_UNROM_512,
    MAPPER_BNROM,
    MAPPER_UN1ROM,
    MAPPER_UXROM_180,
)

# Boards whose mirroring is wired to solder pads: the iNES four-screen bit is
# not implementable, so it falls back to the horizontal/vertical bit.
TWO_BIT_MIRRORING_MAPPERS = frozenset(
    (MAPPER_UXROM, MAPPER_COLOR_DREAMS, MAPPER_UNROM_512, MAPPER_UN1ROM, MAPPER_UXROM_180)
)

MAPPER_NAMES = {
    MAPPER_NROM: "NROM",
    MAPPER_UXROM: "UxROM",
    MAPPER_COLOR_DREAMS: "Color Dreams (UNROM variant)",
    MAPPER_UNROM_512: "UNROM 512",
    MAPPER_BNROM: "BNROM / NINA-001",
    MAPPER_UN1ROM: "UN1ROM",
    MAPPER_UXROM_180: "UxROM (Crazy Climber variant)",
}


def _is_power_of_two(value: int) -> bool:
    return value != 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class INESRom:
    prg_rom_banks: int
    chr_rom_banks: int
    mapper: int
    submapper: int
    has_trainer: bool
    has_battery: bool
    is_nes2: bool
    nametable_arrangement: NametableArrangement
    prg_rom: bytes
    chr_rom: bytes

    @property
    def prg_rom_size(self) -> int:
        return len(self.prg_rom)

    @property
    def chr_rom_size(self) -> int:
        return len(self.chr_rom)

    @property
    def is_nrom(self) -> bool:
        return self.mapper == 0 and self.prg_rom_banks in (1, 2)

    @property
    def is_uxrom(self) -> bool:
        return self.mapper in (
            MAPPER_UXROM,
            MAPPER_COLOR_DREAMS,
            MAPPER_UNROM_512,
            MAPPER_UN1ROM,
            MAPPER_UXROM_180,
        )

    @property
    def has_chr_ram(self) -> bool:
        return self.chr_rom_size == 0

    @property
    def is_supported(self) -> bool:
        """Whether the batched emulator can run this image.

        Mirrors nesle.describe_mapper on the C++ side, which is what the
        bindings actually enforce; keeping both in step means the Python
        error arrives before the CUDA context is created.
        """
        if self.has_trainer:
            return False
        prg = self.prg_rom_size
        if self.mapper == MAPPER_NROM:
            return self.is_nrom
        if self.mapper in (
            MAPPER_UXROM,
            MAPPER_COLOR_DREAMS,
            MAPPER_UNROM_512,
            MAPPER_UN1ROM,
            MAPPER_UXROM_180,
        ):
            return prg >= 0x8000 and prg % 0x4000 == 0
        if self.mapper == MAPPER_BNROM:
            if self.has_chr_ram:
                # BNROM: one 32 KB window, no fixed window.
                return prg >= 0x8000 and prg % 0x8000 == 0
            # NINA-001: 8 KB window, CHR ROM masked by window index.
            chr_size = self.chr_rom_size
            return chr_size >= 0x2000 and _is_power_of_two(chr_size)
        return False

    @property
    def mapper_name(self) -> str:
        return MAPPER_NAMES.get(self.mapper, f"mapper {self.mapper}")

    @property
    def unsupported_reason(self) -> str | None:
        if self.is_supported:
            return None
        reason = (
            f"unsupported mapper: {self.mapper_name} (iNES {self.mapper}); "
            "NeSLE emulates NROM and the UxROM family (0, 2, 11, 30, 34, 94, 180)"
        )
        if self.has_trainer:
            return "ROM trainers are not supported"
        if self.is_nrom is False and self.mapper == MAPPER_NROM:
            return "expected one or two 16 KB PRG ROM banks for NROM"
        return reason

    @property
    def is_supported_mario_target(self) -> bool:
        return (
            self.is_nrom
            and self.submapper == 0
            and self.chr_rom_banks == 1
            and not self.has_trainer
        )


def parse_ines(data: bytes | bytearray | memoryview) -> INESRom:
    raw = bytes(data)
    if len(raw) < HEADER_SIZE:
        raise ValueError("iNES data is shorter than the 16-byte header")
    if raw[:4] != b"NES\x1a":
        raise ValueError("iNES header magic must be NES<EOF>")

    prg_banks = raw[4]
    chr_banks = raw[5]
    flags6 = raw[6]
    flags7 = raw[7]
    is_nes2 = (flags7 & 0x0C) == 0x08
    if is_nes2 and raw[9] != 0:
        raise ValueError("NES 2.0 extended PRG/CHR ROM sizes are not supported yet")

    mapper = (flags6 >> 4) | (flags7 & 0xF0)
    submapper = 0
    if is_nes2:
        mapper |= (raw[8] & 0x0F) << 8
        submapper = raw[8] >> 4
    has_trainer = bool(flags6 & 0x04)

    vertical = bool(flags6 & 0x01)
    four_screen = bool(flags6 & 0x08)
    if mapper == MAPPER_UNROM_512 and four_screen and not vertical:
        arrangement = NametableArrangement.SINGLE_SCREEN_LOWER
    elif mapper in TWO_BIT_MIRRORING_MAPPERS:
        # UxROM boards only have H/V solder pads, so a four-screen bit cannot
        # be honored; fall back to the H/V bit instead of folding the screen.
        arrangement = (
            NametableArrangement.VERTICAL if vertical else NametableArrangement.HORIZONTAL
        )
    elif four_screen:
        arrangement = NametableArrangement.FOUR_SCREEN
    elif vertical:
        arrangement = NametableArrangement.VERTICAL
    else:
        arrangement = NametableArrangement.HORIZONTAL

    prg_size = prg_banks * PRG_BANK_SIZE
    chr_size = chr_banks * CHR_BANK_SIZE
    offset = HEADER_SIZE + (TRAINER_SIZE if has_trainer else 0)
    required = offset + prg_size + chr_size
    if len(raw) < required:
        raise ValueError("iNES data is truncated for declared PRG/CHR sizes")

    prg_rom = raw[offset : offset + prg_size]
    offset += prg_size
    chr_rom = raw[offset : offset + chr_size]

    return INESRom(
        prg_rom_banks=prg_banks,
        chr_rom_banks=chr_banks,
        mapper=mapper,
        submapper=submapper,
        has_trainer=has_trainer,
        has_battery=bool(flags6 & 0x02),
        is_nes2=is_nes2,
        nametable_arrangement=arrangement,
        prg_rom=prg_rom,
        chr_rom=chr_rom,
    )
