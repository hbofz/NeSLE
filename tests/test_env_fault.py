"""Per-env quarantine for an unimplemented opcode, end to end on a real GPU.

cpu::step used to compile to asm("trap;") on the device, which aborts the entire
CUDA launch. One env in a 16k batch reaching a byte the decoder does not know
therefore killed every other env, with an opaque CUDA error and no indication of
which env or which opcode. cpu::step now reports the condition through
StepResult::illegal, console_step_kernel quarantines just that env, and
CudaBatch.faults() says which env died where and on what.

The condition is forced through the public API by patching the program counter in
an FCSX state: poke_ram only reaches RAM, but a state's CPU.PC sub-chunk is just
two bytes, and the ROM's fixed window is full of bytes the decoder rejects. With the
old code this test could not even reach its assertions - the launch would abort - so
"the step returned and the other envs still moved" is the regression.

Skips without a supported ROM or a usable GPU.
"""

from __future__ import annotations

import glob
import struct
import unittest
from pathlib import Path

import numpy as np

# Upstream's CUDA console mode only supports mapper 0 / NROM, so the device half of
# this test has to run on a mapper-0 cartridge or it is testing nothing. Override with
# NESLE_NROM_TEST_ROM.
ROM_DIR = Path(r"C:\games\nes\retro_build\stable-retro-git\tests\roms")
DEFAULT_ROM = "Dr88-FamiconIntro.nes"
FCEUX_STATES = r"C:\games\nes\fceux_rl\curriculum\**\*.fcs"

# The decoder's Illegal set, mirrored from cpu.hpp's decode table. Duplicated rather
# than queried because the table is device-side and there is no host binding for it;
# test_cpu.cpp pins the host half of the same contract.
ILLEGAL_OPCODES = frozenset(
    int(x, 16)
    for x in (
        "02 03 04 07 0B 0C 0F 12 13 14 17 1A 1B 1C 1F 22 23 27 2B 2F 32 33 34 37 3A 3B 3C 3F "
        "42 43 44 47 4B 4F 52 53 54 57 5A 5B 5C 5F 62 63 64 67 6B 6F 72 73 74 77 7A 7B 7C 7F "
        "80 82 83 87 89 8B 8F 92 93 97 9B 9C 9E 9F A3 A7 AB AF B2 B3 B7 BB BF C2 C3 C7 CB CF "
        "D2 D3 D4 D7 DA DB DC DF E2 E3 E7 EB EF F2 F3 F4 F7 FA FB FC FF"
    ).split()
)




def build_minimal_state(pc: int) -> bytes:
    """A minimal but structurally valid legacy FCS state with the PC at `pc`.

    Written from scratch rather than derived from an FCEUX state, so the test needs
    neither a real game nor a saved state on disk: all it wants is for the CPU to
    fetch one byte the decoder rejects. Everything except PC is zero, which is a
    legitimate power-on state - the cartridge's own init never runs, because the very
    first instruction is the illegal one.

    Legacy FCS rather than FCSX on purpose. This is the format the snapshot loader
    already accepts everywhere, so the test does not depend on FCSX parsing - which
    is a separate change and would make this test a referendum on it.
    """
    def sub(body: bytearray, name: bytes, payload: bytes) -> None:
        # A sub-chunk name is exactly four bytes, NUL-padded. "PC" without the
        # padding shifts the whole stream: the reader takes the length from the name's
        # own bytes and then walks off the end of the payload.
        padded = name.ljust(4, b"\x00")[:4]
        body += padded + struct.pack("<I", len(payload)) + payload

    def chunk(out: bytearray, tag: int, body: bytes) -> None:
        out += bytes([tag]) + struct.pack("<I", len(body)) + body

    cpu = bytearray()
    sub(cpu, b"PC", struct.pack("<H", pc))
    for name in (b"A", b"X", b"Y"):
        sub(cpu, name, b"\x00")
    sub(cpu, b"S", b"\xfd")
    sub(cpu, b"P", b"\x24")
    sub(cpu, b"RAM", bytes(2048))

    ppu = bytearray()
    sub(ppu, b"NTAR", bytes(2048))
    sub(ppu, b"PRAM", bytes(32))
    sub(ppu, b"SPRA", bytes(256))
    sub(ppu, b"PPUR", bytes(4))
    sub(ppu, b"RADD", b"\x00\x00")
    sub(ppu, b"TADD", b"\x00\x00")

    # 'FCS\xff' then a 12-byte header; the parser starts chunking at offset 16.
    out = bytearray(b"FCS\xff")
    out += bytes(12)
    chunk(out, 1, bytes(cpu))
    chunk(out, 3, bytes(ppu))
    chunk(out, 16, bytes(8192))
    return bytes(out)


def _first_supported_rom(batch_factory) -> bytes | None:

    """First ROM in the directory the batch will actually accept.

    Chosen by trying to build a batch rather than by asking parse_ines_metadata,
    whose keys differ between revisions; the mapper support this test needs is
    exactly what the constructor checks.
    """
    for path in sorted(glob.glob(str(ROM_DIR / "*.nes"))):
        data = Path(path).read_bytes()
        try:
            batch_factory(data)
        except Exception:
            continue
        return data
    return None


def _first_fceux_state() -> bytes | None:
    for path in sorted(glob.glob(FCEUX_STATES, recursive=True)):
        return Path(path).read_bytes()
    return None


def _patch_pc(image: bytes, pc: int) -> bytes:
    """Return a copy of an FCSX state with CPU.PC replaced.

    Walks the FCSX block/sub-chunk structure and rewrites the two payload bytes of
    the "PC" sub-chunk in the CPU block. Raises if the layout is not what we expect,
    so a format change fails loudly instead of silently patching nothing.
    """
    out = bytearray(image)
    if bytes(out[:4]) != b"FCSX":
        raise AssertionError(f"expected FCSX, got {bytes(out[:4])!r}")
    off = 16
    while off + 5 <= len(out):
        block_id = out[off]
        size = struct.unpack_from("<I", out, off + 1)[0]
        off += 5
        if block_id in (0x01, 0x02):  # CPU / CPU2
            sub = off
            end = off + size
            while sub + 8 <= end:
                name = bytes(out[sub : sub + 4]).rstrip(b"\x00")
                length = struct.unpack_from("<I", out, sub + 4)[0]
                if name == b"PC":
                    if length != 2:
                        raise AssertionError(f"CPU.PC is {length} bytes, expected 2")
                    struct.pack_into("<H", out, sub + 8, pc)
                    return bytes(out)
                sub += 8 + length
        off += size
    raise AssertionError("no CPU.PC sub-chunk in the state")


def _find_quiescent_pc(rom: bytes) -> int | None:
    """An address holding `JMP <self>`: a one-instruction loop that never faults.

    The healthy envs in the isolation test need somewhere to be that is not an illegal
    opcode, and a hand-built state with zeroed RAM cannot provide it - the game's own
    init then walks into filler and faults, which would make the test pass for the
    wrong reason. A JMP-to-self needs no RAM at all. The step kernel even has an idle
    loop fast-forward for 0x4C, so such an env costs almost nothing.
    """
    window = _prg_rom_window(rom)
    for offset in range(len(window) - 2):
        if window[offset] != 0x4C:
            continue
        target = window[offset + 1] | (window[offset + 2] << 8)
        if target == 0xC000 + offset:
            return target
    return None


def _prg_rom_window(rom: bytes) -> bytes:

    """The fixed 16 KiB window of an NROM-family image, i.e. what $C000-$FFFF reads."""
    meta_prg_banks = rom[4]
    prg = rom[16 : 16 + meta_prg_banks * 16 * 1024]
    return prg[-16384:]


def _find_illegal_pc(rom: bytes) -> int | None:
    """An address in the fixed window whose byte the decoder rejects.

    Found from the ROM rather than hard-coded: the filler and padding in an unused
    bank is full of such bytes, and which ones they are moves with the ROM.
    """
    window = _prg_rom_window(rom)
    for offset, byte in enumerate(window):
        if byte in ILLEGAL_OPCODES:
            return 0xC000 + offset
    return None


class EnvFaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from nesle import _cuda_core  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - build dependent
            raise unittest.SkipTest(f"CUDA extension not available: {exc}") from exc
        cls._cuda_core = _cuda_core
        cls.rom = _first_supported_rom(lambda rom: _cuda_core.CudaBatch(2, 1, rom))
        if cls.rom is None:
            raise unittest.SkipTest("no mapper-0 .nes ROM for the device test")

        cls.target_pc = _find_illegal_pc(cls.rom)
        if cls.target_pc is None:
            raise unittest.SkipTest("no unimplemented opcode in the ROM's fixed window")
        cls.patched = build_minimal_state(cls.target_pc)
        # The healthy level needs a PC the decoder accepts and that survives a few
        # frames on its own; a JMP-to-self does both without needing valid RAM.
        cls.quiescent_pc = _find_quiescent_pc(cls.rom)
        if cls.quiescent_pc is None:
            raise unittest.SkipTest("no JMP-to-self in the ROM's fixed window")
        cls.healthy = build_minimal_state(cls.quiescent_pc)
        try:
            probe = _cuda_core.CudaBatch(2, 1, cls.rom)
            del probe
        except Exception as exc:  # pragma: no cover - no GPU
            raise unittest.SkipTest(f"no usable CUDA device: {exc}") from exc

    # ---- the ROM really does have an unimplemented opcode where we point ----

    def test_target_address_holds_an_unimplemented_opcode(self) -> None:
        window = _prg_rom_window(self.rom)
        offset = self.target_pc - 0xC000
        self.assertIn(window[offset], ILLEGAL_OPCODES)
        self.assertEqual(self.patched[:4], b"FCS\xff")
        self.assertEqual(len(self.healthy) - 4, len(self.patched) - 4)

    # The host half of this change is pinned by tests/cpp/test_cpu.cpp, which asserts
    # that step() reports without throwing and step_or_throw() throws with the opcode
    # in hex. It is not repeated here: this branch's NativeConsole predates
    # load_state and its loader is legacy-FCS only.

    # ---- the actual regression ----


    def test_one_faulted_env_does_not_take_the_batch_down(self) -> None:
        """Only the env on the patched level faults; its neighbours keep stepping.

        A snapshot's CPU registers are per *level*, not per env - the reset kernel
        copies snap.pc[level] into every env assigned to that level - so a single
        patched state faults every env that uses it. Two levels with one env on the
        patched one is what actually exercises isolation. Under the old asm("trap;")
        the first step raised a CUDA error and no assertion here was reachable.
        """
        num_envs = 4
        # Level 0 is the patched state, level 1 the untouched one. Env 0 gets the
        # patched level; the rest get the good one.
        batch = self._cuda_core.CudaBatch(
            num_envs,
            1,
            self.rom,
            [self.patched, self.healthy],
            np.array([0, 1, 1, 1], dtype=np.uint8),
        )

        # Zero actions: the fault comes from the patched PC, not the buttons.
        out = batch.step(np.zeros(num_envs, dtype=np.uint8), render_frame=False, copy_obs=False)

        faults = batch.faults()
        self.assertIsInstance(faults, dict)
        self.assertTrue(faults, "the env on the patched level should have been quarantined")
        self.assertLess(len(faults), num_envs, "every env faulted, which is not isolation")
        for env, detail in faults.items():
            self.assertIn(env, range(num_envs))
            pc, opcode = detail
            self.assertEqual(pc, self.target_pc)
            self.assertIn(opcode, ILLEGAL_OPCODES)

        # The launch completed, so the healthy envs are untouched: not marked done,
        # while the faulted one ends its episode cleanly.
        dones = np.asarray(out["dones"])
        for env in range(num_envs):
            if env in faults:
                self.assertEqual(int(dones[env]), 1, "a faulted env must end its episode")
            else:
                self.assertEqual(int(dones[env]), 0, "a healthy env must not be marked done")

    def test_faults_clear_on_reset(self) -> None:
        """A reset env starts from the reset vector, so its old fault is stale."""
        batch = self._cuda_core.CudaBatch(2, 1, self.rom, self.patched)
        batch.step(np.zeros(2, dtype=np.uint8), render_frame=False, copy_obs=False)
        self.assertTrue(batch.faults())

        batch.reset()
        self.assertEqual(
            batch.faults(),
            {},
            "reset must clear the quarantine, or the env stays skipped forever",
        )

    def test_cold_booted_batch_reports_no_faults(self) -> None:
        """A real game booting and running must not trip the quarantine."""
        batch = self._cuda_core.CudaBatch(4, 1, self.rom)
        for _ in range(8):
            batch.step(np.zeros(4, dtype=np.uint8), render_frame=False, copy_obs=False)
        self.assertEqual(batch.faults(), {})


if __name__ == "__main__":
    unittest.main()
