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
    reward_mode: str = "minimal"
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


def _read_bcd(ram, address: int, length: int):
    torch, _, _ = _require_torch()
    value = (ram[:, address] & 0x0F).to(dtype=torch.int32)
    for offset in range(1, length):
        value = value * 10 + (ram[:, address + offset] & 0x0F).to(dtype=torch.int32)
    return value


class TorchSmartMarioReward:
    """GPU RAM reward shaped like mario_rl.rewards.SmartMarioReward for SMB 1-1."""

    x_page = 0x006D
    x_screen = 0x0086
    y_viewport = 0x00B5
    player_state = 0x000E
    lives = 0x075A
    world = 0x075F
    stage = 0x075C
    area = 0x0760
    game_mode = 0x0770
    coins_digits = 0x07ED
    score_digits = 0x07DE

    progress_scale = 0.18
    backtrack_scale = 0.03
    checkpoint_bonus = 3.0
    checkpoint_width = 128.0
    score_scale = 0.04
    coin_bonus = 3.0
    kill_bonus = 8.0
    finish_zone_x = 3100.0
    finish_zone_bonus = 100.0
    flag_zone_x = 3300.0
    flag_bonus = 100.0
    death_penalty = 75.0
    time_penalty = 0.005
    stall_penalty = 0.02
    stall_window = 40
    max_stall_penalty = 0.5
    jump_penalty = 0.005
    neutral_jump_penalty = 0.02
    repeated_jump_penalty = 0.02
    repeated_jump_window = 6
    max_repeated_jump_penalty = 0.25
    left_penalty = 0.01
    bad_button_penalty = 0.25

    def __init__(self, num_envs: int) -> None:
        torch, _, _ = _require_torch()
        device = torch.device("cuda")
        self.last_x = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.max_x = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.checkpoint_index = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.last_score = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.last_coins = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.last_lives = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.last_level = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.stall_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.jump_streak = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.finish_awarded = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.flag_awarded = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ram) -> None:
        self.reset_where(None, ram)

    def reset_where(self, mask, ram) -> None:
        torch, _, _ = _require_torch()
        x = self._x_pos(ram)
        score = self._score(ram).float()
        coins = self._coins(ram)
        lives = ram[:, self.lives].to(torch.int32)
        level = self._level(ram)
        if mask is None:
            self.last_x.copy_(x)
            self.max_x.copy_(x)
            self.checkpoint_index.zero_()
            self.last_score.copy_(score)
            self.last_coins.copy_(coins)
            self.last_lives.copy_(lives)
            self.last_level.copy_(level)
            self.stall_steps.zero_()
            self.jump_streak.zero_()
            self.finish_awarded.zero_()
            self.flag_awarded.zero_()
            return
        self.last_x[mask] = x[mask]
        self.max_x[mask] = x[mask]
        self.checkpoint_index[mask] = 0
        self.last_score[mask] = score[mask]
        self.last_coins[mask] = coins[mask]
        self.last_lives[mask] = lives[mask]
        self.last_level[mask] = level[mask]
        self.stall_steps[mask] = 0
        self.jump_streak[mask] = 0
        self.finish_awarded[mask] = False
        self.flag_awarded[mask] = False

    def compute(self, ram, action_masks, done):
        torch, _, _ = _require_torch()
        x = self._x_pos(ram)
        delta_x = x - self.last_x
        forward = delta_x > 0
        progress = torch.where(
            forward,
            torch.clamp(delta_x, max=16.0) * self.progress_scale,
            torch.clamp(delta_x, min=-16.0) * self.backtrack_scale,
        )
        self.stall_steps = torch.where(
            forward,
            torch.zeros_like(self.stall_steps),
            self.stall_steps + 1,
        )
        self.max_x = torch.maximum(self.max_x, x)
        new_checkpoint_index = torch.floor(self.max_x / self.checkpoint_width).to(torch.int32)
        checkpoint_delta = torch.clamp(new_checkpoint_index - self.checkpoint_index, min=0)
        checkpoint = checkpoint_delta.float() * self.checkpoint_bonus
        self.checkpoint_index = torch.maximum(self.checkpoint_index, new_checkpoint_index)

        score = self._score(ram).float()
        score_delta = torch.clamp(score - self.last_score, min=0.0)
        coins = self._coins(ram)
        coin_delta = coins - self.last_coins
        coin_delta = torch.where(coin_delta < 0, coin_delta + 100, coin_delta)
        coin_delta = torch.clamp(coin_delta, min=0)
        score_reward = score_delta * self.score_scale
        kill_score = score_delta - coin_delta.float() * 200.0
        kill_reward = torch.where(kill_score >= 100.0, torch.full_like(score_reward, self.kill_bonus), torch.zeros_like(score_reward))
        coin_reward = coin_delta.float() * self.coin_bonus

        finish_hit = (~self.finish_awarded) & (self.max_x >= self.finish_zone_x)
        finish_reward = torch.where(finish_hit, torch.full_like(score_reward, self.finish_zone_bonus), torch.zeros_like(score_reward))
        self.finish_awarded |= finish_hit

        is_death = self._is_death(ram)
        flag_hit = (~self.flag_awarded) & (
            (ram[:, self.game_mode] == 2)
            | ((self.max_x >= self.flag_zone_x) & done & (~is_death))
        )
        flag_reward = torch.where(flag_hit, torch.full_like(score_reward, self.flag_bonus), torch.zeros_like(score_reward))
        self.flag_awarded |= flag_hit

        stall_over = torch.clamp((self.stall_steps - self.stall_window + 1).float(), min=0.0)
        stall_reward = -torch.clamp(stall_over * self.stall_penalty, max=self.max_stall_penalty)

        action_reward = self._action_reward(action_masks)
        death_reward = torch.where(done & is_death, torch.full_like(score_reward, -self.death_penalty), torch.zeros_like(score_reward))
        time_reward = torch.full_like(score_reward, -self.time_penalty)

        self.last_x.copy_(x)
        self.last_score.copy_(score)
        self.last_coins.copy_(coins)
        self.last_lives.copy_(ram[:, self.lives].to(torch.int32))
        self.last_level.copy_(self._level(ram))

        return (
            progress
            + checkpoint
            + score_reward
            + kill_reward
            + coin_reward
            + finish_reward
            + flag_reward
            + stall_reward
            + action_reward
            + death_reward
            + time_reward
        )

    def _x_pos(self, ram):
        torch, _, _ = _require_torch()
        return (ram[:, self.x_page].to(torch.float32) * 256.0) + ram[:, self.x_screen].to(torch.float32)

    def _coins(self, ram):
        return _read_bcd(ram, self.coins_digits, 2)

    def _score(self, ram):
        return _read_bcd(ram, self.score_digits, 6)

    def _level(self, ram):
        torch, _, _ = _require_torch()
        return (
            ram[:, self.world].to(torch.int32) * 65536
            + ram[:, self.stage].to(torch.int32) * 256
            + ram[:, self.area].to(torch.int32)
        )

    def _is_death(self, ram):
        state = ram[:, self.player_state]
        return (state == 0x0B) | (state == 0x06) | (ram[:, self.y_viewport] > 1) | (ram[:, self.lives] == 0xFF)

    def _action_reward(self, action_masks):
        torch, _, _ = _require_torch()
        reward = torch.zeros(action_masks.shape[0], dtype=torch.float32, device=action_masks.device)
        jump = (action_masks & 0x01) != 0
        right = (action_masks & 0x80) != 0
        left = (action_masks & 0x40) != 0
        start = (action_masks & 0x08) != 0
        select = (action_masks & 0x04) != 0
        self.jump_streak = torch.where(jump, self.jump_streak + 1, torch.zeros_like(self.jump_streak))
        reward -= jump.float() * self.jump_penalty
        reward -= (jump & (~right)).float() * self.neutral_jump_penalty
        repeated = torch.clamp((self.jump_streak - self.repeated_jump_window).float(), min=0.0)
        reward -= torch.clamp(repeated * self.repeated_jump_penalty, max=self.max_repeated_jump_penalty)
        reward -= left.float() * self.left_penalty
        reward -= (left & right).float() * self.bad_button_penalty
        reward -= (start | select).float() * self.bad_button_penalty
        return reward


def _load_checkpoint(path: str | None, model, optimizer) -> int:
    if path is None:
        return 0
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return 0
    torch, _, _ = _require_torch()
    # weights_only=False is required because we store NumPy RNG state (a numpy ndarray
    # tuple) alongside the tensors. PyTorch 2.6+ defaults to weights_only=True, which
    # disallows arbitrary pickled types. The checkpoints come from this codebase only —
    # we trust them — so explicitly disable the restriction.
    payload = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    # Restore RNG state so a resumed run continues the same sample trajectory. Older
    # checkpoints predate this, so each field is optional and the absence is silently
    # tolerated; new checkpoints round-trip exactly.
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if "cuda_rng_state" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu())
    if "numpy_rng_state" in payload:
        np.random.set_state(payload["numpy_rng_state"])
    return int(payload.get("global_step", 0))


def _save_checkpoint(path: str, model, optimizer, config: NativePPOConfig, global_step: int) -> None:
    torch, _, _ = _require_torch()
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "global_step": int(global_step),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state()
    torch.save(payload, path)


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

    # Cheap config validation BEFORE we open a ROM or talk to CUDA — fails fast on
    # misconfigurations without spending the env setup cost.
    rollout_timesteps = config.num_envs * config.n_steps
    if rollout_timesteps < config.batch_size:
        raise ValueError(
            f"rollout produces {rollout_timesteps} samples per update "
            f"(num_envs={config.num_envs} * n_steps={config.n_steps}) but "
            f"batch_size={config.batch_size} is larger. Either lower --batch-size, "
            f"raise --num-envs, or raise --n-steps."
        )

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
    if action_mask_table.shape[0] != action_dim:
        raise AssertionError(
            f"action mask table length {action_mask_table.shape[0]} does not match "
            f"env.action_space.n={action_dim}; misconfigured action space would index "
            f"GPU memory out of bounds."
        )
    obs = _device_tensor(batch.ram_device())
    rewarder = None
    if config.reward_mode == "smart":
        rewarder = TorchSmartMarioReward(config.num_envs)
        rewarder.reset(obs)
    elif config.reward_mode != "minimal":
        raise ValueError(f"unknown reward mode: {config.reward_mode!r}")
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
        f"reward_mode={config.reward_mode} "
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
                step_out = batch.step_device(action_masks, auto_reset=False, synchronize=True)
                raw_done = _device_tensor(step_out["dones"]).bool()
                ram = _device_tensor(step_out["ram"])
                if rewarder is None:
                    reward = _device_tensor(step_out["rewards"])
                else:
                    reward = rewarder.compute(ram, action_masks, raw_done)
                episode_lengths += 1.0
                if config.max_episode_steps > 0:
                    timeout_done = episode_lengths >= float(config.max_episode_steps)
                    done_bool = raw_done | timeout_done
                else:
                    done_bool = raw_done
                rewards_buf[step].copy_(reward)
                next_done = done_bool.float()

                episode_returns += reward
                if bool(done_bool.any().item()):
                    done_indices = done_bool.nonzero().flatten()
                    recent_returns.extend(episode_returns[done_indices].detach().cpu().tolist())
                    recent_lengths.extend(episode_lengths[done_indices].detach().cpu().tolist())
                    reset_mask = done_bool.detach().to(torch.uint8).cpu().numpy()
                    batch.reset_envs(reset_mask)
                    obs = _device_tensor(batch.ram_device())
                    if rewarder is not None:
                        rewarder.reset_where(done_bool, obs)
                    episode_returns[done_indices] = 0.0
                    episode_lengths[done_indices] = 0.0
                else:
                    obs = ram
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
        choices=["right_only", "simple", "simple_with_start", "complex", "mario", "raw"],
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
    parser.add_argument("--reward-mode", default="minimal", choices=["minimal", "smart"])
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
        reward_mode=args.reward_mode,
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
