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

    if flags6 & 0x08:
        arrangement = NametableArrangement.FOUR_SCREEN
    elif flags6 & 0x01:
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
