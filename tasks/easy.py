"""
tasks/easy.py — Easy task for beginners / fast prototyping.

Configuration:
  - 60-day episode
  - Dry season start (low rainfall variability)
  - Higher starting budget
  - Forgiving reward weights (less penalty)
  - 8 crop slots
  - No severe weather events
"""

from __future__ import annotations
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional

from env.environment import MonsoonFarmEnv
from env.models import FarmAction, FarmObservation, CropStage
from env.reward import RewardFunction
from env.simulator import CROP_CONFIG


class EasyFarmEnv(gym.Env):
    """
    Gymnasium wrapper for the Easy difficulty task.
    
    Features:
    - 60 days, start in December (dry season)
    - Higher budget (₹75,000)
    - Less weather variability
    - Simpler observation (same vector, but easier dynamics)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, seed: int = 42):
        super().__init__()

        reward_fn = RewardFunction(
            profit_weight=1.0,
            yield_weight=0.6,
            eco_weight=0.2,          # less eco pressure
            water_penalty_weight=0.1,
            chemical_penalty_weight=0.15,
            health_bonus_weight=0.15,
            survival_bonus_weight=0.1,
            budget_penalty_weight=0.3,
        )

        self._env = MonsoonFarmEnv(
            num_slots=8,
            episode_length=60,
            start_month=12,          # December — dry, stable
            seed=seed,
            reward_fn=reward_fn,
            initial_budget_inr=75000.0,
            water_tank_capacity=6000.0,
        )

        # Gymnasium spaces
        obs_size = self._env.get_observation_size()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=3.0, shape=(obs_size,), dtype=np.float32
        )

        # Discrete action space: simplified 5-action per slot
        # [0=do_nothing, 1=plant_best_crop, 2=water+feed, 3=pest_control, 4=harvest]
        self.action_space = gym.spaces.MultiDiscrete([5] * self._env.num_slots)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._env.seed = seed
        obs = self._env.reset(seed=seed)
        return self._env.observation_to_numpy(obs), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        farm_action = self._decode_action(action)
        obs, reward, done, info = self._env.step(farm_action)
        return self._env.observation_to_numpy(obs), reward, done, False, info

    def render(self):
        print(self._env.render())

    def _decode_action(self, action: np.ndarray) -> FarmAction:
        """Map discrete MultiDiscrete action to FarmAction."""
        state = self._env.state()
        plant, irrigate, nutrient, pest, harvest = {}, {}, {}, {}, []

        for i, act in enumerate(action):
            slot = state.crop_slots[i]
            act = int(act)

            if act == 0:
                pass  # do nothing
            elif act == 1:
                # Plant best crop for current season
                if slot.stage == int(CropStage.EMPTY):
                    plant[i] = "spinach"  # fast, resilient
                    irrigate[i] = 0.4
                    nutrient[i] = 0.3
            elif act == 2:
                # Water and feed
                irrigate[i] = 0.7
                nutrient[i] = 0.6
                pest[i] = 0
            elif act == 3:
                # Pest control (biological)
                pest[i] = 1
                irrigate[i] = 0.4
                nutrient[i] = 0.3
            elif act == 4:
                # Harvest if ready
                if slot.stage in (int(CropStage.HARVEST), int(CropStage.MATURE)):
                    harvest.append(i)
                    irrigate[i] = 0.0
                else:
                    irrigate[i] = 0.4
                    nutrient[i] = 0.3

        return FarmAction(
            plant_crops=plant,
            irrigation_levels=irrigate,
            nutrient_dose=nutrient,
            pest_control=pest,
            harvest_slots=harvest,
            sell_fraction=0.9,
        )

    def get_wrapped_env(self) -> MonsoonFarmEnv:
        return self._env
