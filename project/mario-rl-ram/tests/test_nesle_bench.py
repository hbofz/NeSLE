from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mario_rl.nesle_bench import (
    NESLE_MARIO_ACTIONS,
    CaseResult,
    build_parser,
    encode_nes_buttons,
    nesle_mario_action_masks,
    parse_csv_ints,
    parse_native_ppo_stdout,
    result_dicts,
    write_csv,
    write_json,
    write_report,
)


def test_nesle_mario_action_masks_match_controller_bits() -> None:
    labels = [label for label, _ in NESLE_MARIO_ACTIONS]
    assert labels == [
        "NOOP",
        "RIGHT",
        "RIGHT_RUN",
        "RIGHT_JUMP",
        "RIGHT_RUN_JUMP",
        "JUMP",
        "RUN_JUMP",
        "LEFT",
        "LEFT_JUMP",
        "DOWN",
        "RIGHT_DOWN",
    ]
    assert nesle_mario_action_masks() == (
        0,
        0x80,
        0x82,
        0x81,
        0x83,
        0x01,
        0x03,
        0x40,
        0x41,
        0x20,
        0xA0,
    )


def test_unknown_button_fails() -> None:
    with pytest.raises(ValueError):
        encode_nes_buttons(["RIGHT", "FIRE"])


def test_parse_csv_ints_validates_values() -> None:
    assert parse_csv_ints("1, 8,32") == (1, 8, 32)
    with pytest.raises(Exception):
        parse_csv_ints("")
    with pytest.raises(Exception):
        parse_csv_ints("1,0")


def test_parser_accepts_all_command_defaults() -> None:
    args = build_parser().parse_args(["all", "--setup", "--run-correctness"])
    assert args.command == "all"
    assert args.setup is True
    assert args.cuda_arch == "sm_80"
    assert "65536" in args.env_counts
    assert "65536" in args.ppo_env_counts
    assert args.ppo_action_space == "mario"


def test_parse_native_ppo_stdout_last_update() -> None:
    stdout = "\n".join(
        [
            "native_ppo backend=cuda-console envs=1024",
            "update=1/2 step=131072 fps=12345 loss=0.1234 clipfrac=0.010 explained_var=0.111 ep_return_100=70.50 ep_len_100=512.0",
            "update=2/2 step=262144 fps=23456 loss=0.0500 clipfrac=0.020 explained_var=0.666 ep_return_100=150.25 ep_len_100=480.0",
        ]
    )
    metrics = parse_native_ppo_stdout(stdout)
    assert metrics["update"] == 2
    assert metrics["updates"] == 2
    assert metrics["fps"] == 23456
    assert metrics["explained_var"] == 0.666
    assert metrics["ep_return_100"] == 150.25


def test_result_writers_emit_json_csv_and_markdown(tmp_path: Path) -> None:
    result = CaseResult(
        suite="env",
        mode="cuda-console-no-copy",
        status="ok",
        num_envs=65536,
        frameskip=4,
        warmup_steps=1,
        timed_steps=2,
        duration_sec=1.0,
        env_steps_per_sec=131072.0,
        frame_steps_per_sec=524288.0,
        gpu_before={"available": True, "gpus": [{"name": "A100", "memory_free_mib": 70000}]},
        gpu_after={"available": True, "gpus": [{"name": "A100", "memory_free_mib": 65000}]},
        extra={"action_mask": 0x80},
    )
    rows = result_dicts([result])

    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"
    report_path = tmp_path / "report.md"
    write_json(json_path, {"results": rows})
    write_csv(csv_path, rows)
    write_report(
        report_path,
        {"mario_rl_sha": "abc", "nesle_sha": "def", "gpu": result.gpu_before},
        rows,
        [],
    )

    assert json.loads(json_path.read_text())["results"][0]["num_envs"] == 65536
    with csv_path.open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["num_envs"] == "65536"
    assert "NeSLE A100 Limits Report" in report_path.read_text()
