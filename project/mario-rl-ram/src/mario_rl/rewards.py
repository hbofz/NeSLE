"""Reward shaping wrappers for Mario RL training.

``InfoRewardShaping`` – lightweight shaping on top of the Stable-Retro base reward.
``SmartMarioReward``  – dense, multi-component reward for serious training.

All parameters are tunable; ``levels.py`` ships per-level presets.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import gymnasium as gym


class InfoRewardShaping(gym.Wrapper):
    """Light shaping from Stable-Retro info variables when they are present."""

    X_KEYS: tuple[Hashable, ...] = (
        "x", "x_pos", "x_position", "screen_x", "scroll_x", "xscroll",
    )

    def __init__(
        self,
        env: gym.Env,
        progress_scale: float = 0.05,
        death_penalty: float = 25.0,
        flag_bonus: float = 100.0,
    ):
        super().__init__(env)
        self.progress_scale = progress_scale
        self.death_penalty = death_penalty
        self.flag_bonus = flag_bonus
        self._last_x: float | None = None

    def reset(self, **kwargs):
        self._last_x = None
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = float(reward)
        x_pos = self._extract_x(info)
        if x_pos is not None:
            if self._last_x is not None:
                delta = max(-5.0, min(5.0, x_pos - self._last_x))
                shaped += self.progress_scale * delta
            self._last_x = x_pos
        if bool(info.get("flag_get", False)):
            shaped += self.flag_bonus
        dead_like = any(bool(info.get(k, False)) for k in ("dead", "death", "gameover"))
        if dead_like or (terminated and not bool(info.get("flag_get", False))):
            shaped -= self.death_penalty
        return obs, shaped, terminated, truncated, info

    def _extract_x(self, info: dict[str, Any]) -> float | None:
        if "xscrollLo" in info and "xscrollHi" in info:
            try:
                return float(info["xscrollLo"]) + 256.0 * float(info["xscrollHi"])
            except (TypeError, ValueError):
                return None
        for key in self.X_KEYS:
            if key in info:
                try:
                    return float(info[key])
                except (TypeError, ValueError):
                    return None
        return None


class SmartMarioReward(gym.Wrapper):
    """Dense Mario reward balancing progress, skill events, and anti-stall pressure.

    Component breakdown (logged in ``info["smart_reward"]`` each step):

    ``progress``    Forward x-scroll reward (+) or backtrack penalty (−).
    ``checkpoint``  One-time bonus each new 128-pixel milestone.
    ``zone``        Optional one-time bonuses for hand-picked level sections.
    ``score``       Proportional reward for score increases (enemies, blocks, powerups).
    ``coin``        Flat bonus per coin collected (handles counter wraparound).
    ``kill``        Flat bonus inferred when score jumps ≥ 100 pts without a coin —
                    almost certainly an enemy stomp or shell hit.  Set ``kill_bonus=0``
                    (default) to disable.
    ``finish``      Large one-time bonuses near the flagpole zone.
    ``level``       Bonus on level transition (redundant with finish in flag-lock mode;
                    set ``level_bonus=0`` in per-level config to disable).
    ``life``        Penalty for losing a life.
    ``death``       Large penalty when episode terminates from death.
    ``time``        Small fixed per-step cost so standing still is never free.
    ``stall``       Increasing pressure when Mario makes no forward progress.
    ``action``      Quality penalties for jump spam, neutral jumps, bad buttons.
    """

    X_KEYS = InfoRewardShaping.X_KEYS

    def __init__(
        self,
        env: gym.Env,
        # Progress
        progress_scale: float = 0.25,
        backtrack_scale: float = 0.05,
        # Score & events
        score_scale: float = 0.025,
        coin_bonus: float = 1.0,
        kill_bonus: float = 0.0,
        max_score_reward: float | None = None,
        max_kill_reward: float | None = None,
        # Milestones
        checkpoint_bonus: float = 5.0,
        checkpoint_width: int = 128,
        zone_bonuses: tuple[tuple[float, float], ...] = (),
        level_bonus: float = 50.0,
        finish_zone_x: float = 3100.0,
        finish_zone_bonus: float = 100.0,
        flag_zone_x: float = 3300.0,
        flag_bonus: float = 100.0,
        # Penalties
        death_penalty: float = 50.0,
        life_loss_penalty: float = 25.0,
        time_penalty: float = 0.01,
        stall_penalty: float = 0.02,
        stall_window: int = 30,
        max_stall_penalty: float = 0.5,
        # Action quality
        jump_penalty: float = 0.03,
        neutral_jump_penalty: float = 0.08,
        repeated_jump_penalty: float = 0.04,
        repeated_jump_window: int = 4,
        max_repeated_jump_penalty: float = 0.5,
        left_penalty: float = 0.02,
        bad_button_penalty: float = 0.25,
    ):
        super().__init__(env)
        self.progress_scale = progress_scale
        self.backtrack_scale = backtrack_scale
        self.score_scale = score_scale
        self.coin_bonus = coin_bonus
        self.kill_bonus = kill_bonus
        self.max_score_reward = max_score_reward
        self.max_kill_reward = max_kill_reward
        self.checkpoint_bonus = checkpoint_bonus
        self.checkpoint_width = checkpoint_width
        self.zone_bonuses = tuple(sorted(zone_bonuses))
        self.level_bonus = level_bonus
        self.finish_zone_x = finish_zone_x
        self.finish_zone_bonus = finish_zone_bonus
        self.flag_zone_x = flag_zone_x
        self.flag_bonus = flag_bonus
        self.death_penalty = death_penalty
        self.life_loss_penalty = life_loss_penalty
        self.time_penalty = time_penalty
        self.stall_penalty = stall_penalty
        self.stall_window = stall_window
        self.max_stall_penalty = max_stall_penalty
        self.jump_penalty = jump_penalty
        self.neutral_jump_penalty = neutral_jump_penalty
        self.repeated_jump_penalty = repeated_jump_penalty
        self.repeated_jump_window = repeated_jump_window
        self.max_repeated_jump_penalty = max_repeated_jump_penalty
        self.left_penalty = left_penalty
        self.bad_button_penalty = bad_button_penalty
        # Episode state
        self._last_x: float | None = None
        self._max_x = 0.0
        self._next_checkpoint = float(checkpoint_width)
        self._last_score: float | None = None
        self._last_coins: int | None = None
        self._last_lives: int | None = None
        self._last_level: tuple[int, int] | None = None
        self._stall_steps = 0
        self._jump_streak = 0
        self._zone_index = 0
        self._score_reward_total = 0.0
        self._kill_reward_total = 0.0
        self._finish_zone_awarded = False
        self._flag_zone_awarded = False

    def reset(self, **kwargs):
        self._last_x = None
        self._max_x = 0.0
        self._next_checkpoint = float(self.checkpoint_width)
        self._last_score = None
        self._last_coins = None
        self._last_lives = None
        self._last_level = None
        self._stall_steps = 0
        self._jump_streak = 0
        self._zone_index = 0
        self._score_reward_total = 0.0
        self._kill_reward_total = 0.0
        self._finish_zone_awarded = False
        self._flag_zone_awarded = False
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        components = {
            "progress": 0.0, "checkpoint": 0.0, "zone": 0.0, "score": 0.0,
            "coin": 0.0, "kill": 0.0, "finish": 0.0, "level": 0.0,
            "life": 0.0, "time": -self.time_penalty, "stall": 0.0,
            "action": 0.0, "death": 0.0,
        }

        # Pre-compute coin delta BEFORE _last_coins is updated by _coin_reward
        coin_delta = self._pending_coin_delta(info)

        # Position & progress
        x_pos = self._extract_x(info)
        if x_pos is not None:
            if self._last_x is not None:
                delta_x = x_pos - self._last_x
                if delta_x > 0:
                    components["progress"] += self.progress_scale * min(delta_x, 16.0)
                    self._stall_steps = 0
                else:
                    components["progress"] += self.backtrack_scale * max(delta_x, -16.0)
                    self._stall_steps += 1
            self._max_x = max(self._max_x, x_pos)
            while self._max_x >= self._next_checkpoint:
                components["checkpoint"] += self.checkpoint_bonus
                self._next_checkpoint += self.checkpoint_width
            components["zone"] = self._zone_reward()
            components["finish"] = self._finish_reward(info, terminated)
            self._last_x = x_pos
        else:
            components["finish"] = self._finish_reward(info, terminated)

        # Score + kill inference (must use self._last_score before _score update)
        components["score"], components["kill"] = self._score_and_kill_reward(info, coin_delta)

        # Remaining components (each updates its own state)
        components["coin"] = self._coin_reward(info)
        components["level"] = self._level_reward(info)
        components["life"] = self._life_reward(info)
        components["action"] = self._action_reward(info)

        if self._stall_steps >= self.stall_window:
            components["stall"] = -min(
                self.max_stall_penalty,
                self.stall_penalty * (self._stall_steps - self.stall_window + 1),
            )
        if terminated and (self._life_like(info) <= -1 or bool(info.get("single_life_done", False))):
            components["death"] = -self.death_penalty

        shaped = sum(components.values())
        info = dict(info)
        info["smart_reward"] = components
        return obs, shaped, terminated, truncated, info

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _extract_x(self, info: dict[str, Any]) -> float | None:
        return InfoRewardShaping._extract_x(self, info)

    def _pending_coin_delta(self, info: dict[str, Any]) -> int:
        """Coins gained this step — computed before _last_coins is updated."""
        if "coins" not in info or self._last_coins is None:
            return 0
        delta = int(info["coins"]) - self._last_coins
        if delta < 0:
            delta += 100  # handle wraparound at 100 coins
        return max(0, delta)

    def _score_and_kill_reward(
        self, info: dict[str, Any], coin_delta: int
    ) -> tuple[float, float]:
        """Return (score_reward, kill_reward).

        Kill inference: score jumps ≥ 100 pts beyond what coins explain
        (each coin gives 200 pts in NES Mario).  Covers stomps, shell hits,
        block-punch kills.  Disabled when ``kill_bonus == 0``.
        """
        if "score" not in info:
            return 0.0, 0.0
        score = float(info["score"])
        score_reward = 0.0
        kill_reward = 0.0
        if self._last_score is not None:
            score_delta = max(0.0, score - self._last_score)
            score_reward = self.score_scale * score_delta
            score_reward = self._apply_reward_cap(
                score_reward,
                "_score_reward_total",
                self.max_score_reward,
            )
            if self.kill_bonus > 0.0:
                kill_score = score_delta - (coin_delta * 200)
                if kill_score >= 100:
                    kill_reward = self.kill_bonus
                    kill_reward = self._apply_reward_cap(
                        kill_reward,
                        "_kill_reward_total",
                        self.max_kill_reward,
                    )
        self._last_score = score
        return score_reward, kill_reward

    def _apply_reward_cap(
        self,
        reward: float,
        total_attr: str,
        cap: float | None,
    ) -> float:
        if reward <= 0.0:
            return 0.0
        current_total = float(getattr(self, total_attr))
        if cap is None:
            setattr(self, total_attr, current_total + reward)
            return reward
        remaining = max(0.0, cap - current_total)
        capped_reward = min(reward, remaining)
        setattr(self, total_attr, current_total + capped_reward)
        return capped_reward

    def _coin_reward(self, info: dict[str, Any]) -> float:
        if "coins" not in info:
            return 0.0
        coins = int(info["coins"])
        reward = 0.0
        if self._last_coins is not None:
            delta = coins - self._last_coins
            if delta < 0:
                delta += 100
            reward = self.coin_bonus * max(0, delta)
        self._last_coins = coins
        return reward

    def _level_reward(self, info: dict[str, Any]) -> float:
        if "levelLo" not in info or "levelHi" not in info:
            return 0.0
        level = (int(info["levelHi"]), int(info["levelLo"]))
        reward = 0.0
        if self._last_level is not None and level != self._last_level:
            reward = self.level_bonus
        self._last_level = level
        return reward

    def _life_reward(self, info: dict[str, Any]) -> float:
        if "lives" not in info:
            return 0.0
        lives = int(info["lives"])
        reward = 0.0
        if self._last_lives is not None and lives < self._last_lives:
            reward = -self.life_loss_penalty * (self._last_lives - lives)
        self._last_lives = lives
        return reward

    def _life_like(self, info: dict[str, Any]) -> int:
        try:
            return int(info.get("lives", 0))
        except (TypeError, ValueError):
            return 0

    def _finish_reward(self, info: dict[str, Any], terminated: bool) -> float:
        reward = 0.0
        if not self._finish_zone_awarded and self._max_x >= self.finish_zone_x:
            reward += self.finish_zone_bonus
            self._finish_zone_awarded = True
        # Real flag detection: prefer flag_get / flag_lock_done / level transition
        # over an x-position guess.  Falls back to flag_zone_x only if no signal.
        if not self._flag_zone_awarded:
            flag_hit = (
                bool(info.get("flag_get", False))
                or bool(info.get("flag_lock_done", False))
                or self._level_changed(info)
            )
            if flag_hit:
                reward += self.flag_bonus
                self._flag_zone_awarded = True
            elif self.flag_zone_x is not None and self._max_x >= self.flag_zone_x:
                # Soft fallback only if the episode also ended normally without
                # a death — avoids paying out when Mario just walked far.
                if terminated and self._life_like(info) >= 0:
                    reward += self.flag_bonus
                    self._flag_zone_awarded = True
        return reward

    def _level_changed(self, info: dict[str, Any]) -> bool:
        if "levelLo" not in info or "levelHi" not in info:
            return False
        try:
            level = (int(info["levelHi"]), int(info["levelLo"]))
        except (TypeError, ValueError):
            return False
        return self._last_level is not None and level != self._last_level

    def _zone_reward(self) -> float:
        reward = 0.0
        while self._zone_index < len(self.zone_bonuses):
            x_pos, bonus = self.zone_bonuses[self._zone_index]
            if self._max_x < x_pos:
                break
            reward += bonus
            self._zone_index += 1
        return reward

    def _action_reward(self, info: dict[str, Any]) -> float:
        buttons = set(info.get("buttons_pressed", ()))
        reward = 0.0
        if "A" in buttons:
            self._jump_streak += 1
            reward -= self.jump_penalty
            if "RIGHT" not in buttons:
                reward -= self.neutral_jump_penalty
            if self._jump_streak > self.repeated_jump_window:
                reward -= min(
                    self.max_repeated_jump_penalty,
                    self.repeated_jump_penalty * (self._jump_streak - self.repeated_jump_window),
                )
        else:
            self._jump_streak = 0
        if "LEFT" in buttons:
            reward -= self.left_penalty
        if "LEFT" in buttons and "RIGHT" in buttons:
            reward -= self.bad_button_penalty
        if "START" in buttons or "SELECT" in buttons:
            reward -= self.bad_button_penalty
        return reward
