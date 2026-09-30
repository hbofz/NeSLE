r"""Report the iNES header of every ROM given, so new test carts can be vetted
without trusting the filename.

    python scripts/report_rom_mappers.py C:/games/nes
    python scripts/report_rom_mappers.py C:/games/nes/Crazy\ Climber\ \(J\).nes

The mapper number is the thing under test here, and it lives only in the
header: a translation of a mapper 94 or 180 board is often re-tagged as plain
mapper 2, because both are the same UNROM PCB with different logic gates. So
check the reported mapper, not the file name. CRC32 is printed alongside it to
pin down the exact revision.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nesle.rom import parse_ines  # noqa: E402


def describe(path: Path) -> str:
    data = path.read_bytes()
    crc = f"{zlib.crc32(data) & 0xFFFFFFFF:08X}"
    try:
        rom = parse_ines(data)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the sweep
        return f"{path.name}: not a usable iNES image ({exc}) crc32={crc}"

    chr_kind = f"{rom.chr_rom_banks * 8}K ROM" if rom.chr_rom_banks else "none (CHR RAM)"
    extras = []
    if rom.has_chr_ram:
        extras.append("chr_ram")
    if rom.has_battery:
        extras.append("battery")
    if rom.has_trainer:
        extras.append("trainer")
    if rom.is_nes2:
        extras.append(f"nes2/sub{rom.submapper}")
    tail = (" [" + ", ".join(extras) + "]") if extras else ""

    verdict = "supported" if rom.is_supported else f"UNSUPPORTED ({rom.unsupported_reason})"
    return (
        f"{path.name}\n"
        f"    crc32={crc}  mapper={rom.mapper} ({rom.mapper_name}){tail}\n"
        f"    prg={rom.prg_rom_banks * 16}K  chr={chr_kind}  "
        f"mirroring={rom.nametable_arrangement.name.lower()}\n"
        f"    {verdict}"
    )


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2

    paths: list[Path] = []
    for arg in args:
        p = Path(arg)
        paths.extend(sorted(p.glob("*.nes")) if p.is_dir() else [p])

    for path in paths:
        print(describe(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
