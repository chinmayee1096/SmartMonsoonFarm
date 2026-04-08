"""
tasks/hard.py — Hard difficulty task.

Configuration:
  - 120-day episode
  - Starts in May (pre-monsoon heatwaves + monsoon transition)
  - 16 slots but lower starting budget (₹30,000)
  - Market adversarial (higher price volatility)
  - Stricter eco penalties
  - Agent must manage crop diversity for market resilience
"""

from __future__ import annotations
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional

from env.environment import MonsoonFarmEnv
from env.models import FarmAction, FarmObservation, CropStage
from env.reward import RewardFunction

VALID_CROPS = ["spinach", "lettuce", "tomato", "herbs"]


class HardFarmEnv(gym.Env):
    """
    Hard difficulty: longer horizon, harsher weather, tighter constraints.

    The agent must:
    - Survive heatwaves (May-June) before monsoon brings relief
    - Manage cash flow on a tight budget
    - Maintain eco-score for bonus rewards
    - Make sell-vs-hold decisions under volatile prices

    Action space: same Box structure as Medium (scalable)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, seed: int = 42):
        super().__init__()

        # Stricter reward weights
        reward_fn = RewardFunction(
            profit_weight=1.2,
            yield_weight=0.4,
            eco_weight=0.5,          # higher eco pressure
            water_penalty_weight=0.4,
            chemical_penalty_weight=0.5,
            health_bonus_weight=0.08,
            survival_bonus_weight=0.3,
            budget_penalty_weight=0.8,
            profit_scale=8000.0,     # harder to get high profit reward
        )

        self._env = MonsoonFarmEnv(
            num_slots=16,
            episode_length=120,
            start_month=5,           # May — pre-monsoon heatwaves
            seed=seed,
            reward_fn=reward_fn,
            initial_budget_inr=30000.0,
            water_tank_capacity=4000.0,   # smaller tank
        )
        self.num_slots = self._env.num_slots

        obs_size = self._env.get_observation_size()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=3.0, shape=(obs_size,), dtype=np.float32
        )

        # Same continuous action structure as Medium
        n_slot_dims = self.num_slots * 5
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(n_slot_dims + 1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs = self._env.reset(seed=seed)
        return self._env.observation_to_numpy(obs), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        farm_action = self._decode_action(action)
        obs, reward, done, info = self._env.step(farm_action)
        return self._env.observation_to_numpy(obs), reward, done, False, info

    def render(self):
        print(self._env.render())

    def _decode_action(self, action: np.ndarray) -> FarmAction:
        state = self._env.state()
        plant, irrigate, nutrient, pest_ctrl, harvest = {}, {}, {}, {}, []

        for i in range(self.num_slots):
            slot = state.crop_slots[i]
            base = i * 5

            irr  = float(np.clip(action[base + 0], 0.0, 1.0))
            nut  = float(np.clip(action[base + 1], 0.0, 1.0))
            pest = float(np.clip(action[base + 2], 0.0, 1.0))
            harv = float(np.clip(action[base + 3], 0.0, 1.0))
            plnt = float(np.clip(action[base + 4], 0.0, 1.0))

            pest_discrete = 0 if pest < 0.33 else (1 if pest < 0.66 else 2)

            if slot.stage == int(CropStage.EMPTY):
                # More selective planting in hard mode
                if plnt > 0.5:
                    # Preference: tomato (high yield) or herbs (high price)
                    if plnt > 0.75:
                        plant[i] = "herbs"
                    elif plnt > 0.6:
                        plant[i] = "tomato"
                    else:
                        crop_idx = min(int(plnt * len(VALID_CROPS)), len(VALID_CROPS) - 1)
                        plant[i] = VALID_CROPS[crop_idx]
            else:
                irrigate[i] = irr
                nutrient[i] = nut
                pest_ctrl[i] = pest_discrete

                if harv > 0.55 and slot.stage in (int(CropStage.HARVEST), int(CropStage.MATURE)):
                    harvest.append(i)

        sell_frac = float(np.clip(action[-1], 0.0, 1.0))

        return FarmAction(
            plant_crops=plant,
            irrigation_levels=irrigate,
            nutrient_dose=nutrient,
            pest_control=pest_ctrl,
            harvest_slots=harvest,
            sell_fraction=sell_frac,
        )

    def get_wrapped_env(self) -> MonsoonFarmEnv:
        return self._env
