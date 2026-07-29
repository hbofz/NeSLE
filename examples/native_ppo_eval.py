"""Evaluate a native PPO checkpoint: episode stats and an optional gameplay GIF.

Loads a checkpoint produced by ``nesle.native_ppo`` / ``examples/native_ppo_train.py``,
runs the policy in a small vectorized env, reports where Mario actually gets
(max x-position, flags, episode lengths), and can record env 0 to a GIF.

Example:

    python examples/native_ppo_eval.py "Super Mario Bros. (World).nes" \
        --checkpoint checkpoints/native_ppo_smart_1050ti.pt \
        --gif-out docs/assets/agent-1-1.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import nesle
from nesle.native_ppo import _make_model, _require_cuda_torch

# SMB RAM addresses (see nesle.smb).
X_PAGE = 0x006D
X_SCREEN = 0x0086
GAME_MODE = 0x0770


def record_best_episode(args, payload, hidden_size: int, action_space: str) -> None:
    """Run N episodes in one frameskip-1 env, keep the best, write a smooth GIF.

    The env steps single NES frames so every frame can be rendered; the policy
    still chooses an action once per ``--frameskip`` frames, matching the
    cadence it was trained with. Playback is real time (every 2nd frame at
    ~30 fps).
    """
    torch, _, Categorical = _require_cuda_torch()

    import nesle

    env = nesle.make_vec(
        rom_path=args.rom_path,
        num_envs=1,
        backend="cuda",
        observation_mode="ram",
        action_space=action_space,
        frameskip=1,
        reset_state_path=args.state,
    )
    model = _make_model(2048, int(env.action_space.n), hidden_size)
    model.load_state_dict(payload["model"])
    model.eval()

    max_frames_per_ep = args.max_steps * args.frameskip
    best_frames: list[np.ndarray] = []
    best_x = -1
    obs = env.reset()
    for episode in range(args.record_best):
        frames: list[np.ndarray] = []
        ep_x = 0
        done = False
        while not done and len(frames) < max_frames_per_ep:
            with torch.no_grad():
                logits, _ = model(torch.as_tensor(obs, device="cuda"))
                action = int(Categorical(logits=logits).sample()[0])
            for _ in range(args.frameskip):
                obs, _, dones, _ = env.step([action])
                frames.append(env.render()[0].copy())
                x = int(obs[0, X_PAGE]) * 256 + int(obs[0, X_SCREEN])
                ep_x = max(ep_x, x)
                if dones[0]:
                    done = True
                    break
        print(f"episode {episode + 1}/{args.record_best}: max_x={ep_x} frames={len(frames)}")
        if ep_x > best_x:
            best_x = ep_x
            best_frames = frames
        # env auto-resets on done; if the episode hit the frame cap, force one.
        if not done:
            obs = env.reset()

    from PIL import Image

    out = Path(args.gif_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in best_frames[::2]]  # 60 fps source -> 30 fps
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=33,  # ~30 fps == real time for every-2nd-frame of 60 fps
        loop=0,
        optimize=True,
    )
    print(f"best episode: max_x={best_x} -> {out} ({len(images)} frames, {out.stat().st_size / 1e6:.1f} MB)")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rom_path")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state", default="docs/data/smb_level1_1.state")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true", help="argmax actions instead of sampling")
    parser.add_argument("--gif-out", default=None, help="write a GIF of env 0 to this path")
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument(
        "--record-best",
        type=int,
        default=0,
        metavar="N",
        help="smooth recording mode: run N episodes in a single frameskip-1 env "
        "(policy still acts every --frameskip frames, as in training), render "
        "every NES frame, and write only the best episode (highest x) to "
        "--gif-out at real-time speed",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch, _, Categorical = _require_cuda_torch()
    torch.manual_seed(args.seed)

    payload = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    config = payload.get("config", {})
    hidden_size = int(config.get("hidden_size", 256))
    action_space = config.get("action_space", "mario")

    if args.record_best > 0:
        if args.gif_out is None:
            parser.error("--record-best requires --gif-out")
        record_best_episode(args, payload, hidden_size, action_space)
        return

    env = nesle.make_vec(
        rom_path=args.rom_path,
        num_envs=args.num_envs,
        backend="cuda",
        observation_mode="ram",
        action_space=action_space,
        frameskip=args.frameskip,
        reset_state_path=args.state,
    )
    action_dim = int(env.action_space.n)

    model = _make_model(2048, action_dim, hidden_size)
    model.load_state_dict(payload["model"])
    model.eval()
    trained_steps = int(payload.get("global_step", 0))
    print(
        f"checkpoint={args.checkpoint} trained_steps={trained_steps} "
        f"action_space={action_space} hidden={hidden_size} envs={args.num_envs} "
        f"policy={'argmax' if args.deterministic else 'sample'}"
    )

    obs = env.reset()
    n = args.num_envs
    ep_len = np.zeros(n, dtype=np.int64)
    ep_max_x = np.zeros(n, dtype=np.int64)
    prev_flag = np.zeros(n, dtype=bool)
    finished_lens: list[int] = []
    finished_max_x: list[int] = []
    flag_events = 0
    frames: list[np.ndarray] = []

    for _ in range(args.max_steps):
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(obs, device="cuda"))
            if args.deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = Categorical(logits=logits).sample()
        obs, _, dones, infos = env.step(actions.cpu().numpy())

        if args.gif_out is not None:
            frames.append(env.render()[0].copy())

        x = obs[:, X_PAGE].astype(np.int64) * 256 + obs[:, X_SCREEN].astype(np.int64)
        flag = obs[:, GAME_MODE] == 2
        flag_events += int(np.count_nonzero(flag & ~prev_flag))
        prev_flag = flag
        ep_len += 1
        ep_max_x = np.maximum(ep_max_x, x)

        for i, done in enumerate(dones):
            if done:
                term = infos[i].get("terminal_observation")
                if term is not None:
                    tx = int(term[X_PAGE]) * 256 + int(term[X_SCREEN])
                    ep_max_x[i] = max(ep_max_x[i], tx)
                finished_lens.append(int(ep_len[i]))
                finished_max_x.append(int(ep_max_x[i]))
                ep_len[i] = 0
                ep_max_x[i] = 0
                prev_flag[i] = False

    if finished_lens:
        print(
            f"episodes={len(finished_lens)} "
            f"mean_len={np.mean(finished_lens):.0f} "
            f"mean_max_x={np.mean(finished_max_x):.0f} "
            f"best_max_x={max(finished_max_x)} "
            f"flag_events={flag_events}"
        )
        print("(SMB 1-1 flag pole is at x~3175; best_max_x near that means the level was cleared)")
    else:
        print(f"no episode finished within {args.max_steps} steps; flag_events={flag_events}")

    if args.gif_out is not None:
        from PIL import Image

        out = Path(args.gif_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(f) for f in frames]
        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=max(20, int(1000 / args.gif_fps)),
            loop=0,
            optimize=True,
        )
        print(f"gif: {out} ({len(images)} frames, {out.stat().st_size / 1e6:.1f} MB)")

    env.close()


if __name__ == "__main__":
    main()
