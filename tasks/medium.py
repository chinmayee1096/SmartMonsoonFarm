"""
tasks/medium.py — Medium difficulty task.

Configuration:
  - 90-day episode starting in June (full monsoon exposure)
  - 12 crop slots
  - Standard budget (₹50,000)
  - Full weather variability including occasional heatwaves
  - Continuous action space for SB3 SAC/PPO
"""

from __future__ import annotations
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional

from env.environment import MonsoonFarmEnv
from env.models import FarmAction, FarmObservation, CropStage
from env.reward import RewardFunction
from env.simulator import CROP_CONFIG

VALID_CROPS = ["spinach", "lettuce", "tomato", "herbs"]


class MediumFarmEnv(gym.Env):
    """
    Gymnasium wrapper for Medium difficulty.
    Uses a continuous Box action space compatible with PPO / SAC.

    Action vector (per slot, flattened):
      [irrigate_frac, nutrient_frac, pest_type_continuous, harvest_flag] * num_slots
      + [plant_decision_per_slot (0-3 crop index)] * num_slots
      + [sell_fraction]

    Total action dims: num_slots * 4 + num_slots + 1 = 12*5+1 = 61
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, seed: int = 42):
        super().__init__()

        reward_fn = RewardFunction(
            profit_weight=1.0,
            yield_weight=0.5,
            eco_weight=0.3,
            water_penalty_weight=0.2,
            chemical_penalty_weight=0.3,
            health_bonus_weight=0.1,
            survival_bonus_weight=0.2,
            budget_penalty_weight=0.5,
        )

        self._env = MonsoonFarmEnv(
            num_slots=12,
            episode_length=90,
            start_month=6,   # June (monsoon)
            seed=seed,
            reward_fn=reward_fn,
            initial_budget_inr=50000.0,
            water_tank_capacity=5000.0,
        )
        self.num_slots = self._env.num_slots

        obs_size = self._env.get_observation_size()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=3.0, shape=(obs_size,), dtype=np.float32
        )

        # Continuous action space
        # Per slot: [irrigate(0-1), nutrient(0-1), pest(0-1 maps to 0/1/2), harvest(0-1)]
        # Per slot: [plant(0-1 maps to 0-3 crop index, only used if slot empty)]
        # Global: [sell_fraction(0-1)]
        n_slot_dims = self.num_slots * 5  # 4 continuous + 1 plant
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
        """Decode continuous action vector to FarmAction."""
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

            # Map pest continuous to discrete {0,1,2}
            pest_discrete = 0 if pest < 0.33 else (1 if pest < 0.66 else 2)

            if slot.stage == int(CropStage.EMPTY):
                # Plant if plnt > 0.4
                if plnt > 0.4:
                    crop_idx = min(int(plnt * len(VALID_CROPS)), len(VALID_CROPS) - 1)
                    plant[i] = VALID_CROPS[crop_idx]
            else:
                irrigate[i] = irr
                nutrient[i] = nut
                pest_ctrl[i] = pest_discrete

                if harv > 0.6 and slot.stage in (int(CropStage.HARVEST), int(CropStage.MATURE)):
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
