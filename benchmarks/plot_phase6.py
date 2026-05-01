from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PALETTE = {
    "rgb": "#2f6f9f",
    "ram_obs": "#7952b3",
    "render_only": "#d8842f",
    "no_copy": "#3f8f5f",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def _fmt(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _svg_bar_chart(rows: list[dict[str, Any]], title: str, output: Path) -> None:
    if not rows:
        return
    width = 960
    height = 520
    margin_left = 96
    margin_bottom = 84
    margin_top = 72
    plot_width = width - margin_left - 48
    plot_height = height - margin_top - margin_bottom
    max_value = max(float(row["env_steps_per_sec"]) for row in rows)
    labels = [f"{row['mode']} @ {row['num_envs']}" for row in rows]
    bar_gap = 8
    bar_width = max(12, (plot_width - bar_gap * (len(rows) - 1)) / max(len(rows), 1))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#202020">{title}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 36}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
    ]
    for tick in range(5):
        value = max_value * tick / 4
        y = height - margin_bottom - (value / max_value) * plot_height if max_value else height - margin_bottom
        parts.append(f'<line x1="{margin_left - 6}" y1="{y:.1f}" x2="{width - 36}" y2="{y:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="12" text-anchor="end" fill="#555">{_fmt(value)}</text>')

    for index, row in enumerate(rows):
        value = float(row["env_steps_per_sec"])
        x = margin_left + index * (bar_width + bar_gap)
        bar_height = (value / max_value) * plot_height if max_value else 0
        y = height - margin_bottom - bar_height
        color = PALETTE.get(str(row["mode"]), "#555")
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#222">{_fmt(value)}</text>')
        label = labels[index]
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - margin_bottom + 18}" '
            'font-family="Arial, sans-serif" font-size="11" text-anchor="end" '
            f'transform="rotate(-35 {x + bar_width / 2:.1f} {height - margin_bottom + 18})" fill="#333">{label}</text>'
        )

    parts.append(f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - 16}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle" fill="#444">CUDA-console env-steps/sec, A100</text>')
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 6 SVG plots.")
    parser.add_argument("--input", type=Path, default=Path("docs/data/phase6-a100-summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()

    rows = _load_rows(args.input)
    fs4 = [row for row in rows if row["frameskip"] == 4 and row["num_envs"] in {1, 8, 32, 128}]
    copy_gap = [row for row in fs4 if row["mode"] in {"rgb", "ram_obs", "no_copy"}]
    _svg_bar_chart(copy_gap, "Phase 6 CUDA-Console Copy Gap", args.output_dir / "phase6-copy-gap.svg")

    frameskip_rows = [
        row
        for row in rows
        if row["num_envs"] == 32 and row["mode"] == "no_copy" and row["frameskip"] in {1, 2, 4, 8}
    ]
    if len(frameskip_rows) == 4:
        _svg_bar_chart(frameskip_rows, "Phase 6 Frameskip Ablation", args.output_dir / "phase6-frameskip.svg")
    print(f"wrote SVG plots to {args.output_dir}")


if __name__ == "__main__":
    main()
