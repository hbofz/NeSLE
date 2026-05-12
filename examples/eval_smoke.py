"""Load a trained PPO model and roll out one episode, printing Mario's progress.

This is the visceral "does the agent actually play SMB?" check. It also doubles as a
working example of inference loop on the cuda-console backend with snapshot reset.
"""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

import nesle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rom", default="Super Mario Bros. (World).nes")
    p.add_argument("--model", default="nesle_ppo_smoke3")
    p.add_argument("--state", default="docs/data/smb_level1_1.state")
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()

    env = nesle.make_vec(
        rom_path=args.rom,
        num_envs=1,
        backend="cuda",
        observation_mode="ram",
        action_space="simple",
        reset_state_path=args.state,
    )
    model = PPO.load(args.model, env=env, device="cpu")
    print(f"loaded {args.model}.zip; rolling out {args.steps} steps on cuda-console + snapshot reset")

    obs = env.reset()
    ram0 = env._cuda_batch.ram()[0]
    x_start = int(ram0[0x006D]) * 0x100 + int(ram0[0x0086])
    total_reward = 0.0
    max_x = x_start
    episode_lengths = []
    episode_rewards = []
    current_ep_len = 0
    current_ep_reward = 0.0

    SIMPLE_LABELS = ["NOOP", "right", "right+A", "right+B", "right+A+B", "A", "left"]
    action_histogram = np.zeros(7, dtype=np.int64)

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(action)
        action_histogram[int(action[0])] += 1
        total_reward += float(rewards[0])
        current_ep_reward += float(rewards[0])
        current_ep_len += 1
        ram = env._cuda_batch.ram()[0]
        x = int(ram[0x006D]) * 0x100 + int(ram[0x0086])
        max_x = max(max_x, x)
        if step % 20 == 0 or dones[0]:
            print(f"  step {step:3d}  action={SIMPLE_LABELS[int(action[0])]:9s}  "
                  f"x_pos={x:4d}  reward={float(rewards[0]):+5.1f}  done={bool(dones[0])}")
        if dones[0]:
            episode_lengths.append(current_ep_len)
            episode_rewards.append(current_ep_reward)
            current_ep_len = 0
            current_ep_reward = 0.0
    if current_ep_len > 0:
        episode_lengths.append(current_ep_len)
        episode_rewards.append(current_ep_reward)

    print()
    print(f"=== Summary ===")
    print(f"  total reward:    {total_reward:+.1f}")
    print(f"  max x_pos:       {max_x} (start was {x_start}, advanced {max_x - x_start})")
    print(f"  episodes:        {len(episode_lengths)} (lengths={episode_lengths}, rewards=[{', '.join(f'{r:+.1f}' for r in episode_rewards)}])")
    print(f"  action histogram:")
    for label, count in zip(SIMPLE_LABELS, action_histogram, strict=True):
        bar = "#" * int(40 * count / max(1, int(action_histogram.sum())))
        print(f"    {label:10s} {int(count):4d}  {bar}")
    env.close()


if __name__ == "__main__":
    main()
