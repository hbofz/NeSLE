from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import nesle


@dataclass
class NativePPOConfig:
    rom_path: str
    num_envs: int = 1024
    total_timesteps: int = 1_000_000
    n_steps: int = 128
    batch_size: int = 8192
    update_epochs: int = 4
    frameskip: int = 4
    action_space: str = "simple_with_start"
    reset_state_path: str | None = None
    reset_state_paths: tuple[str, ...] = ()
    max_episode_steps: int = 0
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 256
    seed: int = 1
    checkpoint_path: str = "nesle_native_ppo.pt"
    log_interval: int = 1
    progress_bar: bool = False


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.distributions.categorical import Categorical
    except ImportError as exc:  # pragma: no cover - depends on optional rl deps
        raise SystemExit(
            "Install PyTorch to run native PPO. For the usual setup: pip install -e '.[rl]'"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("native PPO requires a CUDA-enabled PyTorch build.")
    return torch, nn, Categorical


def _orthogonal_init(module, gain: float) -> None:
    torch, _, _ = _require_torch()
    if hasattr(module, "weight"):
        torch.nn.init.orthogonal_(module.weight, gain)
    if hasattr(module, "bias") and module.bias is not None:
        torch.nn.init.constant_(module.bias, 0.0)


def _make_model(obs_dim: int, action_dim: int, hidden_size: int):
    torch, nn, _ = _require_torch()

    class RamActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.actor = nn.Linear(hidden_size, action_dim)
            self.critic = nn.Linear(hidden_size, 1)
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    _orthogonal_init(layer, np.sqrt(2.0))
            _orthogonal_init(self.actor, 0.01)
            _orthogonal_init(self.critic, 1.0)

        def forward(self, obs):
            x = obs.float().div_(255.0)
            features = self.net(x)
            return self.actor(features), self.critic(features).squeeze(-1)

    return RamActorCritic().to(torch.device("cuda"))


def _device_tensor(view):
    torch, _, _ = _require_torch()
    if hasattr(view, "__dlpack__"):
        return torch.utils.dlpack.from_dlpack(view)
    return torch.as_tensor(view, device=torch.device("cuda"))


def _load_checkpoint(path: str | None, model, optimizer) -> int:
    if path is None:
        return 0
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return 0
    torch, _, _ = _require_torch()
    payload = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("global_step", 0))


def _save_checkpoint(path: str, model, optimizer, config: NativePPOConfig, global_step: int) -> None:
    torch, _, _ = _require_torch()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "global_step": int(global_step),
        },
        path,
    )


def _make_env(config: NativePPOConfig):
    reset_paths: Sequence[str] | None = config.reset_state_paths or None
    env = nesle.make_vec(
        rom_path=config.rom_path,
        num_envs=config.num_envs,
        frameskip=config.frameskip,
        action_space=config.action_space,
        backend="cuda",
        observation_mode="ram",
        reset_state_path=config.reset_state_path,
        reset_state_paths=reset_paths,
        max_episode_steps=config.max_episode_steps,
    )
    batch = getattr(env, "_cuda_batch", None)
    if batch is None or str(batch.name) != "cuda-console":
        raise SystemExit("native PPO requires the CUDA console backend. Build nesle._cuda_core first.")
    if not hasattr(batch, "step_device") or not hasattr(batch, "ram_device"):
        raise SystemExit(
            "nesle._cuda_core was built without the device tensor bridge. Rebuild the CUDA extension."
        )
    batch.reset_device()
    return env, batch


def train_native_ppo(config: NativePPOConfig, resume_from: str | None = None) -> None:
    torch, _, Categorical = _require_torch()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    env, batch = _make_env(config)
    obs_dim = 2048
    action_dim = int(env.action_space.n)
    model = _make_model(obs_dim, action_dim, config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    global_step = _load_checkpoint(resume_from, model, optimizer)

    action_mask_table = torch.tensor(tuple(env.action_masks), dtype=torch.uint8, device="cuda")
    obs = _device_tensor(batch.ram_device())
    rollout_timesteps = config.num_envs * config.n_steps
    num_updates = max(1, math.ceil(config.total_timesteps / rollout_timesteps))
    planned_timesteps = num_updates * rollout_timesteps

    obs_buf = torch.empty((config.n_steps, config.num_envs, obs_dim), dtype=torch.uint8, device="cuda")
    actions_buf = torch.empty((config.n_steps, config.num_envs), dtype=torch.long, device="cuda")
    logprobs_buf = torch.empty((config.n_steps, config.num_envs), dtype=torch.float32, device="cuda")
    rewards_buf = torch.empty((config.n_steps, config.num_envs), dtype=torch.float32, device="cuda")
    dones_buf = torch.empty((config.n_steps, config.num_envs), dtype=torch.float32, device="cuda")
    values_buf = torch.empty((config.n_steps, config.num_envs), dtype=torch.float32, device="cuda")

    episode_returns = torch.zeros(config.num_envs, dtype=torch.float32, device="cuda")
    episode_lengths = torch.zeros(config.num_envs, dtype=torch.float32, device="cuda")
    recent_returns: list[float] = []
    recent_lengths: list[float] = []
    next_done = torch.zeros(config.num_envs, dtype=torch.float32, device="cuda")
    start_time = time.monotonic()

    print(
        "native_ppo "
        f"backend={batch.name} envs={config.num_envs} n_steps={config.n_steps} "
        f"batch={config.batch_size} action_space={config.action_space} "
        f"target_steps={config.total_timesteps} planned_steps={planned_timesteps}"
    )

    progress = _make_progress_bar(config.progress_bar, planned_timesteps, global_step)
    try:
        for update in range(1, num_updates + 1):
            for step in range(config.n_steps):
                global_step += config.num_envs
                obs_buf[step].copy_(obs)
                dones_buf[step].copy_(next_done)
                with torch.no_grad():
                    logits, value = model(obs)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                actions_buf[step].copy_(action)
                logprobs_buf[step].copy_(logprob)
                values_buf[step].copy_(value)

                action_masks = action_mask_table[action].contiguous()
                step_out = batch.step_device(action_masks, auto_reset=True, synchronize=True)
                reward = _device_tensor(step_out["rewards"])
                done_u8 = _device_tensor(step_out["dones"])
                rewards_buf[step].copy_(reward)
                next_done = done_u8.float()

                episode_returns += reward
                episode_lengths += 1.0
                if bool(done_u8.any().item()):
                    done_indices = done_u8.nonzero().flatten()
                    recent_returns.extend(episode_returns[done_indices].detach().cpu().tolist())
                    recent_lengths.extend(episode_lengths[done_indices].detach().cpu().tolist())
                    episode_returns[done_indices] = 0.0
                    episode_lengths[done_indices] = 0.0

                obs = _device_tensor(step_out["ram"])
                if progress is not None:
                    progress.update(config.num_envs)

            with torch.no_grad():
                _, next_value = model(obs)
                advantages, returns = compute_gae(
                    rewards_buf,
                    dones_buf,
                    values_buf,
                    next_done,
                    next_value,
                    config.gamma,
                    config.gae_lambda,
                )

            b_obs = obs_buf.reshape((-1, obs_dim))
            b_actions = actions_buf.reshape(-1)
            b_logprobs = logprobs_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)
            batch_count = b_obs.shape[0]

            clipfracs: list[float] = []
            losses: list[float] = []
            for _ in range(config.update_epochs):
                permutation = torch.randperm(batch_count, device="cuda")
                for start in range(0, batch_count, config.batch_size):
                    mb_inds = permutation[start : start + config.batch_size]
                    logits, newvalue = model(b_obs[mb_inds])
                    dist = Categorical(logits=logits)
                    newlogprob = dist.log_prob(b_actions[mb_inds])
                    entropy = dist.entropy().mean()
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    value_loss = 0.5 * (newvalue - b_returns[mb_inds]).pow(2).mean()
                    entropy_loss = entropy
                    loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * value_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        clipfracs.append(
                            ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                        )
                        losses.append(loss.item())

            if update % max(1, config.log_interval) == 0:
                elapsed = max(1e-6, time.monotonic() - start_time)
                fps = int(global_step / elapsed)
                recent_slice = recent_returns[-100:]
                recent_len_slice = recent_lengths[-100:]
                ep_ret = float(np.mean(recent_slice)) if recent_slice else float("nan")
                ep_len = float(np.mean(recent_len_slice)) if recent_len_slice else float("nan")
                explained_var = _explained_variance(b_values, b_returns)
                if progress is not None:
                    progress.set_postfix(
                        fps=fps,
                        ret=f"{ep_ret:.1f}",
                        ev=f"{explained_var:.2f}",
                        clip=f"{np.mean(clipfracs):.2f}",
                    )
                print(
                    f"update={update}/{num_updates} step={global_step} fps={fps} "
                    f"loss={np.mean(losses):.4f} clipfrac={np.mean(clipfracs):.3f} "
                    f"explained_var={explained_var:.3f} ep_return_100={ep_ret:.2f} ep_len_100={ep_len:.1f}",
                    flush=True,
                )
                _save_checkpoint(config.checkpoint_path, model, optimizer, config, global_step)
    finally:
        if progress is not None:
            progress.close()

    _save_checkpoint(config.checkpoint_path, model, optimizer, config, global_step)
    env.close()


def _make_progress_bar(enabled: bool, total: int, initial: int):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print("progress bar requested but tqdm is not installed; falling back to update logs")
        return None
    return tqdm(total=total, initial=min(initial, total), desc="native PPO", unit="steps")


def _explained_variance(values, returns) -> float:
    torch, _, _ = _require_torch()
    with torch.no_grad():
        y_var = torch.var(returns)
        if float(y_var.item()) == 0.0:
            return float("nan")
        return float((1.0 - torch.var(returns - values) / y_var).item())


def compute_gae(rewards, dones, values, next_done, next_value, gamma: float, gae_lambda: float):
    torch, _, _ = _require_torch()
    advantages = torch.zeros_like(rewards, device=rewards.device)
    lastgaelam = torch.zeros(rewards.shape[1], dtype=torch.float32, device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_nonterminal = 1.0 - next_done
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages, advantages + values


def parse_args() -> tuple[NativePPOConfig, str | None]:
    parser = argparse.ArgumentParser(description="Train NeSLE with the CUDA-native PPO path.")
    parser.add_argument("rom_path", help="Path to Super Mario Bros. (World).nes")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument(
        "--action-space",
        default="simple_with_start",
        choices=["right_only", "simple", "simple_with_start", "complex", "raw"],
    )
    parser.add_argument("--reset-state-path", default=None)
    parser.add_argument("--reset-state-paths", nargs="+", default=())
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-path", default="nesle_native_ppo.pt")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()
    config = NativePPOConfig(
        rom_path=args.rom_path,
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        update_epochs=args.update_epochs,
        frameskip=args.frameskip,
        action_space=args.action_space,
        reset_state_path=args.reset_state_path,
        reset_state_paths=tuple(args.reset_state_paths),
        max_episode_steps=args.max_episode_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_size=args.hidden_size,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        log_interval=args.log_interval,
        progress_bar=args.progress_bar,
    )
    if config.reset_state_path and config.reset_state_paths:
        raise SystemExit("Pass either --reset-state-path or --reset-state-paths, not both.")
    return config, args.resume_from


def main() -> None:
    config, resume_from = parse_args()
    train_native_ppo(config, resume_from=resume_from)


if __name__ == "__main__":
    main()
