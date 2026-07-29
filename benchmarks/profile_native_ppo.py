"""Break down where native-PPO rollout time goes, and test a CUDA-graph policy.

Measures, at a given env count:
  1. raw emulator stepping (step_device, constant actions)
  2. rollout loop = stepping + policy inference + sampling (eager)
  3. same rollout loop with the policy forward captured in a CUDA graph

The gap between (1) and (2) is the per-step cost of inference + action
plumbing; (3) shows whether graph capture recovers any of it. Compare with the
fps of a real training run (which adds reward, GAE, and the learn phase).

    python benchmarks/profile_native_ppo.py "Super Mario Bros. (World).nes" --num-envs 2048
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nesle.native_ppo import NativePPOConfig, _device_tensor, _make_env, _make_model, _require_cuda_torch


def timed(label: str, steps: int, num_envs: int, fn, torch) -> float:
    fn()  # warmup
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    per_step_ms = elapsed / steps * 1000
    env_steps_s = steps * num_envs / elapsed
    print(f"{label:34s} {per_step_ms:8.2f} ms/batched-step   {env_steps_s:12,.0f} env-steps/s")
    return per_step_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rom_path")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--reset-state-path", default="docs/data/smb_level1_1.state")
    args = parser.parse_args()

    torch, _, Categorical = _require_cuda_torch()
    torch.manual_seed(0)

    config = NativePPOConfig(
        rom_path=args.rom_path,
        num_envs=args.num_envs,
        action_space="mario",
        reset_state_path=args.reset_state_path,
    )
    env, batch = _make_env(config)
    action_dim = int(env.action_space.n)
    model = _make_model(2048, action_dim, args.hidden_size)
    model.eval()
    mask_table = torch.tensor(tuple(env.action_masks), dtype=torch.uint8, device="cuda")

    obs = _device_tensor(batch.ram_device())
    const_actions = torch.zeros(args.num_envs, dtype=torch.uint8, device="cuda")

    print(f"envs={args.num_envs} steps={args.steps} hidden={args.hidden_size}")

    # 1. raw stepping
    def raw_step():
        batch.step_device(const_actions, auto_reset=False, synchronize=True)

    t_raw = timed("raw step_device", args.steps, args.num_envs, raw_step, torch)

    # 2. eager rollout step (inference + sample + mask lookup + step)
    def eager_rollout():
        with torch.no_grad():
            logits, _ = model(obs)
            actions = Categorical(logits=logits).sample()
        batch.step_device(mask_table[actions].contiguous(), auto_reset=False, synchronize=True)

    t_eager = timed("rollout step (eager policy)", args.steps, args.num_envs, eager_rollout, torch)

    # 3. rollout step with CUDA-graphed policy forward
    static_obs = torch.zeros_like(obs, dtype=torch.float32)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad():
        # warmup on a side stream, as the capture API requires
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model.net(static_obs)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(graph):
            feats = model.net(static_obs)
            static_logits = model.actor(feats)

    def graphed_rollout():
        static_obs.copy_(obs)
        static_obs.div_(255.0)
        graph.replay()
        with torch.no_grad():
            actions = Categorical(logits=static_logits).sample()
        batch.step_device(mask_table[actions].contiguous(), auto_reset=False, synchronize=True)

    t_graph = timed("rollout step (graphed policy)", args.steps, args.num_envs, graphed_rollout, torch)

    print()
    print(f"inference overhead (eager):   {t_eager - t_raw:6.2f} ms/step "
          f"({(t_eager - t_raw) / t_eager * 100:.0f}% of rollout step)")
    print(f"inference overhead (graphed): {t_graph - t_raw:6.2f} ms/step")
    if t_graph < t_eager:
        print(f"CUDA graph speedup on rollout: {t_eager / t_graph:.2f}x")
    else:
        print("CUDA graph did not help at this configuration")

    env.close()


if __name__ == "__main__":
    main()
