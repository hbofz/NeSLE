from __future__ import annotations

import argparse
import platform
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local Mario RL setup.")
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"machine={platform.machine()}")

    try:
        import stable_retro
        import stable_baselines3
        import torch
    except ImportError as exc:
        raise SystemExit(f"Import failed: {exc}") from exc

    print("stable_retro=import ok")
    print(f"stable_baselines3={stable_baselines3.__version__}")
    print(f"torch={torch.__version__}")

    games = stable_retro.data.list_games()
    print(f"known_games={len(games)}")
    print(f"{args.game}_known={args.game in games}")

    try:
        rom_path = stable_retro.data.get_romfile_path(args.game)
    except FileNotFoundError:
        print(f"{args.game}_rom=missing")
        print("next_step=python -m stable_retro.import roms/")
    else:
        print(f"{args.game}_rom={rom_path}")


if __name__ == "__main__":
    main()
