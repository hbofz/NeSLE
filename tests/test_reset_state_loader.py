"""Reading reset states from disk: needs no GPU, so this runs on CI.

docs/data/smb_level1_1_fceux.fc1 is a real save, written by FCEUX 2.6.5 with no
compression, part way through World 1-1. Compressed saves are rebuilt from it the way FCEUX does, with zlib's
compress2 at the default level (src/state.cpp, FCEUSS_SaveMS).
"""
from __future__ import annotations

import gzip
import struct
import tracemalloc
import zlib
from pathlib import Path

import pytest

from nesle import env as nesle_env
from nesle.env import NesleVecEnv, _load_reset_state

DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
REAL_SAVE = (DATA / "smb_level1_1_fceux.fc1").read_bytes()
BODY = REAL_SAVE[16:]
LEGACY = b"FCS\xff" + bytes(12) + BODY
STORED = 0xFFFFFFFF
LIMIT = 16 * 1024 * 1024


def _fcsx(total: int, compressed_size: int, payload: bytes) -> bytes:
    return b"FCSX" + struct.pack("<III", total, 20605, compressed_size) + payload


def _compressed(body: bytes = BODY, level: int = -1) -> bytes:
    payload = zlib.compress(body, level)
    return _fcsx(len(body), len(payload), payload)


def _load(tmp_path: Path, data: bytes) -> bytes:
    path = tmp_path / "state"
    path.write_bytes(data)
    return _load_reset_state(path)


def _rom(tmp_path: Path) -> str:
    rom = tmp_path / "rom.nes"
    rom.write_bytes(b"NES\x1a" + bytes([2, 1, 0, 0]) + bytes(8) + bytes(32 * 1024) + bytes(8 * 1024))
    return str(rom)


def test_the_fixture_really_is_a_stored_fcsx_save():
    assert REAL_SAVE[:4] == b"FCSX"
    assert struct.unpack_from("<III", REAL_SAVE, 4) == (len(BODY), 20605, STORED)
    # Tag 31 and the 64 KB tag 8 are written by FCEUX and absent from Stable Retro states.
    assert b"\x1f" + struct.pack("<I", 369) in BODY
    assert b"\x08" + struct.pack("<I", 65536) in BODY


def test_stored_save_becomes_the_chunk_stream_the_parser_reads(tmp_path):
    assert _load_reset_state(DATA / "smb_level1_1_fceux.fc1") == LEGACY
    assert _load(tmp_path, REAL_SAVE) == LEGACY


@pytest.mark.parametrize("level", [-1, *range(10)])
def test_compressed_save_gives_the_same_stream(tmp_path, level):
    assert _load(tmp_path, _compressed(level=level)) == LEGACY


def test_gzip_wrapped_fcsx_is_unwrapped_twice(tmp_path):
    assert _load(tmp_path, gzip.compress(_compressed())) == LEGACY


def test_stable_retro_states_still_load():
    for path in sorted(DATA.glob("*.state")):
        assert _load_reset_state(path) == gzip.decompress(path.read_bytes())


def test_uncompressed_legacy_state_passes_through(tmp_path):
    assert _load(tmp_path, LEGACY) == LEGACY


@pytest.mark.parametrize("data", [
    pytest.param(REAL_SAVE[:15], id="header-cut-short"),
    pytest.param(_fcsx(0, STORED, BODY), id="stored-size-field-0"),
    pytest.param(_fcsx(100, STORED, BODY), id="stored-size-field-too-small"),
    pytest.param(_fcsx(len(BODY) + 1, STORED, BODY), id="stored-size-field-too-large"),
    pytest.param(_fcsx(len(BODY) - 1, 0, b""), id="compressed-and-empty"),
])
def test_rejects_a_bad_container(tmp_path, data):
    with pytest.raises(ValueError):
        _load(tmp_path, data)


def test_rejects_a_compressed_size_that_disagrees_with_the_file(tmp_path):
    good = _compressed()
    with pytest.raises(ValueError):
        _load(tmp_path, good + b"\0")
    with pytest.raises(ValueError):
        _load(tmp_path, good[:-1])


def test_rejects_a_declared_size_that_disagrees_with_the_stream(tmp_path):
    payload = zlib.compress(BODY)
    with pytest.raises(ValueError):
        _load(tmp_path, _fcsx(len(BODY) - 1, len(payload), payload))
    with pytest.raises(ValueError):
        _load(tmp_path, _fcsx(len(BODY) + 1, len(payload), payload))


def test_rejects_corrupt_and_truncated_streams(tmp_path):
    good = _compressed()
    flipped = bytearray(good)
    flipped[len(good) // 2] ^= 0x40
    wrong_checksum = bytearray(good)
    wrong_checksum[-1] ^= 0x01
    short_payload = zlib.compress(BODY)[:-10]
    for data in (bytes(flipped), bytes(wrong_checksum), _fcsx(len(BODY), len(short_payload), short_payload)):
        with pytest.raises(ValueError):
            _load(tmp_path, data)


def test_rejects_truncated_gzip(tmp_path):
    with pytest.raises(ValueError):
        _load(tmp_path, gzip.compress(LEGACY)[:-10])


@pytest.mark.parametrize("wrap", [
    lambda bomb: _fcsx(16, len(zlib.compress(bomb, 9)), zlib.compress(bomb, 9)),
    lambda bomb: _fcsx(len(bomb), len(zlib.compress(bomb, 9)), zlib.compress(bomb, 9)),
    lambda bomb: gzip.compress(bomb, 9),
], ids=["fcsx-small-header", "fcsx-honest-header", "gzip"])
def test_decompression_bomb_is_stopped_early(tmp_path, wrap):
    data = wrap(bytes(LIMIT * 8))
    tracemalloc.start()
    try:
        with pytest.raises(ValueError):
            _load(tmp_path, data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < LIMIT * 3


def test_rejects_a_declared_size_above_the_limit(tmp_path):
    payload = zlib.compress(b"\0" * 16)
    with pytest.raises(ValueError):
        _load(tmp_path, _fcsx(0xFFFFFFF0, len(payload), payload))
    with pytest.raises(ValueError):
        _load(tmp_path, _fcsx(LIMIT + 1, STORED, bytes(LIMIT + 1)))


@pytest.mark.parametrize("wrap", [
    lambda body: _fcsx(len(body), len(zlib.compress(body)), zlib.compress(body)),
    lambda body: gzip.compress(body),
], ids=["fcsx", "gzip"])
def test_limit_is_exact_on_the_inflate_paths(tmp_path, wrap):
    assert _load(tmp_path, wrap(b"FCS\xff" + bytes(LIMIT - 4)))[:4] == b"FCS\xff"
    with pytest.raises(ValueError):
        _load(tmp_path, wrap(bytes(LIMIT + 1)))


def test_rejects_a_compressed_size_larger_than_the_file(tmp_path):
    payload = zlib.compress(BODY)
    with pytest.raises(ValueError):
        _load(tmp_path, _fcsx(len(BODY), len(payload) + 1, payload))


def test_rejects_trailing_bytes_after_gzip(tmp_path):
    with pytest.raises(ValueError):
        _load(tmp_path, gzip.compress(LEGACY) + b"junk")


def test_error_names_the_file(tmp_path):
    with pytest.raises(ValueError, match="broken.fc1"):
        path = tmp_path / "broken.fc1"
        path.write_bytes(_fcsx(0, STORED, b"x"))
        _load_reset_state(path)


def test_auto_backend_does_not_silently_drop_a_requested_state(tmp_path, monkeypatch):
    # A rejected state used to fall through to a power-on env, ignoring the request.
    def rejected(*args):
        raise RuntimeError("FCS: no CPU section")

    monkeypatch.setattr(nesle_env, "_make_cuda_batch", rejected)
    with pytest.raises(RuntimeError, match="no CPU section"):
        NesleVecEnv(_rom(tmp_path), num_envs=1, backend="auto", device="cuda",
                    reset_state_path=DATA / "smb_level1_1_fceux.fc1")


def test_auto_backend_still_falls_back_when_no_state_was_requested(tmp_path, monkeypatch):
    def unavailable(*args):
        raise RuntimeError("no GPU")

    monkeypatch.setattr(nesle_env, "_make_cuda_batch", unavailable)
    env = NesleVecEnv(_rom(tmp_path), num_envs=1, backend="auto", device="cuda")
    assert env._cuda_batch is None
    env.close()
