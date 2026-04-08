"""
reward.py — Multi-objective reward function for the monsoon farm RL environment.

Reward components:
  1. Profit reward  — revenue from crop sales
  2. Yield reward   — kg of produce harvested
  3. Eco bonus      — reward for maintaining good eco-score
  4. Water penalty  — penalise over-irrigation
  5. Chemical penalty — penalise excessive pesticide use
  6. Partial progress — small daily rewards for healthy crops (dense signal)
  7. Survival bonus  — bonus for keeping crops alive through disasters
"""

from __future__ import annotations
from dataclasses import dataclass
from env.models import FarmState, FarmAction, CropStage
from env.simulator import CROP_CONFIG


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components for logging."""
    profit_reward: float = 0.0
    yield_reward: float = 0.0
    eco_bonus: float = 0.0
    water_penalty: float = 0.0
    chemical_penalty: float = 0.0
    daily_health_bonus: float = 0.0
    survival_bonus: float = 0.0
    budget_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "profit_reward":    round(self.profit_reward, 4),
            "yield_reward":     round(self.yield_reward, 4),
            "eco_bonus":        round(self.eco_bonus, 4),
            "water_penalty":    round(self.water_penalty, 4),
            "chemical_penalty": round(self.chemical_penalty, 4),
            "daily_health_bonus": round(self.daily_health_bonus, 4),
            "survival_bonus":   round(self.survival_bonus, 4),
            "budget_penalty":   round(self.budget_penalty, 4),
            "total":            round(self.total, 4),
        }


class RewardFunction:
    """
    Configurable multi-objective reward function.
    Weights can be adjusted per task difficulty.
    """

    def __init__(
        self,
        profit_weight: float = 1.0,
        yield_weight: float = 0.5,
        eco_weight: float = 0.3,
        water_penalty_weight: float = 0.2,
        chemical_penalty_weight: float = 0.3,
        health_bonus_weight: float = 0.1,
        survival_bonus_weight: float = 0.2,
        budget_penalty_weight: float = 0.5,
        profit_scale: float = 10000.0,   # INR normalisation factor
    ):
        self.profit_weight = profit_weight
        self.yield_weight = yield_weight
        self.eco_weight = eco_weight
        self.water_penalty_weight = water_penalty_weight
        self.chemical_penalty_weight = chemical_penalty_weight
        self.health_bonus_weight = health_bonus_weight
        self.survival_bonus_weight = survival_bonus_weight
        self.budget_penalty_weight = budget_penalty_weight
        self.profit_scale = profit_scale

    def compute(
        self,
        prev_state: FarmState,
        action: FarmAction,
        next_state: FarmState,
        step_revenue_inr: float,
        step_cost_inr: float,
        harvested_kg: float,
        water_used_today_l: float,
        pesticide_used_today: float,
    ) -> RewardBreakdown:
        rb = RewardBreakdown()

        # ---------------------------------------------------------------
        # 1. Profit reward (normalised to ~[-1, 1])
        # ---------------------------------------------------------------
        step_profit = step_revenue_inr - step_cost_inr
        rb.profit_reward = self.profit_weight * (step_profit / self.profit_scale)

        # ---------------------------------------------------------------
        # 2. Yield reward (per kg harvested, encourages actual production)
        # ---------------------------------------------------------------
        rb.yield_reward = self.yield_weight * (harvested_kg / 5.0)  # 5 kg/day = excellent

        # ---------------------------------------------------------------
        # 3. Eco bonus (reward for maintaining good eco-score)
        # ---------------------------------------------------------------
        eco = next_state.eco.eco_score
        rb.eco_bonus = self.eco_weight * max(0.0, eco - 0.5) * 2.0  # 0 when eco<0.5

        # ---------------------------------------------------------------
        # 4. Water penalty (over-irrigation wastes resources)
        # ---------------------------------------------------------------
        tank_capacity = next_state.resources.water_tank_capacity
        # Maximum sensible daily use: 1.5L * num_slots
        max_sensible_water = 1.5 * next_state.num_slots
        if water_used_today_l > max_sensible_water * 1.5:
            overuse_fraction = (water_used_today_l - max_sensible_water * 1.5) / (tank_capacity * 0.1)
            rb.water_penalty = -self.water_penalty_weight * min(1.0, overuse_fraction)

        # Also penalise if tank runs dry
        if next_state.resources.water_tank_liters < 100:
            rb.water_penalty -= self.water_penalty_weight * 0.5

        # ---------------------------------------------------------------
        # 5. Chemical penalty (pesticide overuse)
        # ---------------------------------------------------------------
        if pesticide_used_today > 0:
            # Penalise proportionally; biological control (cheaper) is less penalised
            chemical_slots = sum(
                1 for v in action.pest_control.values() if v == 2
            )
            rb.chemical_penalty = -self.chemical_penalty_weight * (chemical_slots / max(next_state.num_slots, 1))

            # Additional penalty if runoff risk is high
            if next_state.eco.runoff_risk > 0.5:
                rb.chemical_penalty -= self.chemical_penalty_weight * 0.3

        # ---------------------------------------------------------------
        # 6. Daily health bonus (dense signal to encourage good farming)
        # ---------------------------------------------------------------
        active_slots = [
            s for s in next_state.crop_slots
            if s.stage not in (int(CropStage.EMPTY), int(CropStage.DEAD))
        ]
        if active_slots:
            avg_health = sum(s.health for s in active_slots) / len(active_slots)
            rb.daily_health_bonus = self.health_bonus_weight * avg_health * 0.1

        # ---------------------------------------------------------------
        # 7. Survival bonus (crops surviving heatwave / heavy rain)
        # ---------------------------------------------------------------
        from env.models import WeatherType
        if next_state.weather.weather_type in (
            int(WeatherType.HEATWAVE), int(WeatherType.HEAVY_RAIN)
        ):
            surviving = sum(
                1 for s in next_state.crop_slots
                if s.health > 0.5 and s.stage not in (int(CropStage.EMPTY), int(CropStage.DEAD))
            )
            rb.survival_bonus = self.survival_bonus_weight * (surviving / max(next_state.num_slots, 1)) * 0.2

        # ---------------------------------------------------------------
        # 8. Budget bankruptcy penalty
        # ---------------------------------------------------------------
        if next_state.resources.budget_inr < 0:
            rb.budget_penalty = -self.budget_penalty_weight

        # ---------------------------------------------------------------
        # Total
        # ---------------------------------------------------------------
        rb.total = (
            rb.profit_reward +
            rb.yield_reward +
            rb.eco_bonus +
            rb.water_penalty +
            rb.chemical_penalty +
            rb.daily_health_bonus +
            rb.survival_bonus +
            rb.budget_penalty
        )

        return rb
