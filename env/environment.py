"""
environment.py — OpenEnv-compatible Smart Monsoon-Resilient Hydroponic Farm.

Implements the full OpenEnv API:
  - reset()  → initial FarmState
  - step(action) → (observation, reward, done, info)
  - state()  → current FarmState

Time step: 1 day
Episode length: 60-120 days (configurable)
"""

from __future__ import annotations
import random
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

from env.models import (
    FarmState, FarmAction, FarmObservation,
    CropSlot, WeatherState, ResourceState, MarketState, EcoMetrics,
    CropStage, WeatherType,
)
from env.simulator import (
    WeatherSimulator, CropSimulator, MarketSimulator,
    EcoSimulator, CROP_CONFIG,
)
from env.reward import RewardFunction, RewardBreakdown


class MonsoonFarmEnv:
    """
    OpenEnv-compatible Hydroponic Farm environment.

    Observation space: continuous vector of ~94 floats (see FarmObservation)
    Action space: composite dict-based (plant, irrigate, fertilise, pest, harvest)

    For Stable-Baselines3 compatibility, see wrappers in tasks/*.py.
    """

    VALID_CROPS = ["spinach", "lettuce", "tomato", "herbs"]

    def __init__(
        self,
        num_slots: int = 12,
        episode_length: int = 90,
        start_month: int = 6,    # June (monsoon onset)
        seed: int = 42,
        reward_fn: Optional[RewardFunction] = None,
        initial_budget_inr: float = 50000.0,
        water_tank_capacity: float = 5000.0,
        task_config: Optional[Dict] = None,
    ):
        self.num_slots = num_slots
        self.episode_length = episode_length
        self.seed = seed
        self.start_month = start_month

        # Day-of-year for month start
        month_doy = {1:1,2:32,3:60,4:91,5:121,6:152,7:182,8:213,9:244,10:274,11:305,12:335}
        self.start_doy = month_doy.get(start_month, 152)

        self.reward_fn = reward_fn or RewardFunction()
        self.initial_budget = initial_budget_inr
        self.water_tank_capacity = water_tank_capacity
        self.task_config = task_config or {}

        # Simulators (seeded)
        self._weather_sim = WeatherSimulator(start_day=self.start_doy, seed=seed)
        self._market_sim  = MarketSimulator(seed=seed)
        self._crop_sim    = CropSimulator()
        self._rng         = random.Random(seed)

        # Internal state
        self._state: Optional[FarmState] = None
        self._held_stock: Dict[str, float] = {}  # crop_type -> kg held for better price
        self._episode_rewards: List[float] = []

        # Observation vector size
        self.obs_size = 22 + num_slots * 6

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> FarmObservation:
        """Reset environment to initial state. Returns first observation."""
        if seed is not None:
            self.seed = seed
            self._weather_sim = WeatherSimulator(start_day=self.start_doy, seed=seed)
            self._market_sim  = MarketSimulator(seed=seed)
            self._rng         = random.Random(seed)

        # Build initial crop slots (all empty)
        slots = [
            CropSlot(slot_id=i)
            for i in range(self.num_slots)
        ]

        # Initial weather (warm, pre-monsoon or monsoon depending on start month)
        initial_weather = self._weather_sim.simulate_day(
            WeatherState(), episode_day=0
        )

        self._state = FarmState(
            day=0,
            episode_length=self.episode_length,
            num_slots=self.num_slots,
            crop_slots=slots,
            weather=initial_weather,
            resources=ResourceState(
                water_tank_liters=self.water_tank_capacity * 0.8,
                water_tank_capacity=self.water_tank_capacity,
                nutrient_stock_units=200.0,
                pesticide_stock_units=50.0,
                budget_inr=self.initial_budget,
            ),
            market=MarketState(),
            eco=EcoMetrics(eco_score=1.0),
        )

        self._held_stock = {c: 0.0 for c in self.VALID_CROPS}
        self._episode_rewards = []

        return self._build_observation()

    def step(self, action: FarmAction) -> Tuple[FarmObservation, float, bool, Dict]:
        """
        Advance one day.
        Returns (observation, reward, done, info).
        """
        assert self._state is not None, "Call reset() before step()"
        prev_state = self._state
        day = prev_state.day + 1

        # ---- 1. Apply planting actions ----
        step_cost = 0.0
        for slot_id, crop_type in action.plant_crops.items():
            if 0 <= slot_id < self.num_slots:
                slot = prev_state.crop_slots[slot_id]
                if (
                    slot.stage == int(CropStage.EMPTY)
                    and crop_type in self.VALID_CROPS
                ):
                    cfg = CROP_CONFIG[crop_type]
                    slot.crop_type = crop_type
                    slot.stage = int(CropStage.SEEDING)
                    slot.days_since_planted = 0
                    slot.health = 1.0
                    slot.nutrient_level = 0.5
                    slot.water_stress = 0.0
                    slot.pest_pressure = 0.0
                    step_cost += cfg["seed_cost_inr"]

        # ---- 2. Simulate weather ----
        new_weather = self._weather_sim.simulate_day(prev_state.weather, day)

        # ---- 3. Update market ----
        new_market = self._market_sim.step(prev_state.market, day)

        # ---- 4. Process each crop slot ----
        water_used_total = 0.0
        pesticide_used_total = 0.0
        fertilizer_used_total = 0.0
        updated_slots = []

        for slot in prev_state.crop_slots:
            sid = slot.slot_id
            if slot.stage in (int(CropStage.EMPTY), int(CropStage.DEAD)):
                updated_slots.append(slot)
                continue

            # Irrigation
            irrigate_frac = action.irrigation_levels.get(sid, 0.3)
            irrigate_frac = max(0.0, min(1.0, irrigate_frac))
            cfg = CROP_CONFIG.get(slot.crop_type, {})
            max_water_l = cfg.get("water_per_day_l", 1.0) * 1.5
            water_l = irrigate_frac * max_water_l
            water_used_total += water_l

            # Nutrient dosing
            nutrient_frac = action.nutrient_dose.get(sid, 0.3)
            nutrient_frac = max(0.0, min(1.0, nutrient_frac))
            max_nutrient = cfg.get("nutrient_per_day", 0.5) * 1.5
            nutrient_given = nutrient_frac * max_nutrient
            fertilizer_used_total += nutrient_given
            step_cost += nutrient_given * 5.0  # INR per unit nutrient

            # Pest control
            pest_action = action.pest_control.get(sid, 0)
            if pest_action == 1:  # biological
                pest_cost = 20.0
                pesticide_given = 0.5
                step_cost += pest_cost
            elif pest_action == 2:  # chemical
                pest_cost = 10.0
                pesticide_given = 2.0
                step_cost += pest_cost
                pesticide_used_total += pesticide_given
            else:
                pesticide_given = 0.0

            # Update pest pressure
            new_pest = self._crop_sim.update_pest_pressure(slot, new_weather, pest_action, self._rng)

            # Update water stress
            new_water_stress = self._crop_sim.compute_water_stress(slot, water_l, new_weather)

            # Update health
            new_health = self._crop_sim.compute_health(slot, new_water_stress, nutrient_given, new_weather)

            # Advance growth days
            new_days = slot.days_since_planted + 1
            new_stage = int(self._crop_sim.stage_for_days(slot.crop_type, new_days))

            # Death condition
            if new_health <= 0.05:
                new_stage = int(CropStage.DEAD)

            # Update expected yield
            temp_slot = CropSlot(
                slot_id=sid,
                crop_type=slot.crop_type,
                stage=new_stage,
                days_since_planted=new_days,
                health=new_health,
                water_stress=new_water_stress,
                pest_pressure=new_pest,
            )
            expected_yield = self._crop_sim.compute_expected_yield(temp_slot)
            temp_slot.expected_yield_kg = expected_yield

            updated_slots.append(temp_slot)

        # ---- 5. Harvest actions ----
        harvested_kg = 0.0
        step_revenue = 0.0

        for slot_id in action.harvest_slots:
            if 0 <= slot_id < self.num_slots:
                slot = updated_slots[slot_id]
                if slot.stage in (int(CropStage.HARVEST), int(CropStage.MATURE)):
                    yield_kg = slot.expected_yield_kg
                    crop_type = slot.crop_type

                    # Determine how much to sell now vs hold
                    sell_now_kg = yield_kg * action.sell_fraction
                    hold_kg = yield_kg * (1.0 - action.sell_fraction)
                    self._held_stock[crop_type] = self._held_stock.get(crop_type, 0.0) + hold_kg

                    # Revenue from immediate sale
                    price = self._market_sim.get_price(new_market, crop_type)
                    step_revenue += sell_now_kg * price
                    harvested_kg += yield_kg

                    # Reset slot to empty
                    updated_slots[slot_id] = CropSlot(slot_id=slot_id)

                    # Harvest labour cost
                    step_cost += 50.0  # INR per harvest event

        # Sell held stock (partial, to avoid indefinite holding)
        for crop_type, kg in list(self._held_stock.items()):
            if kg > 0.1 and self._rng.random() < 0.3:
                price = self._market_sim.get_price(new_market, crop_type)
                sell_kg = min(kg, kg * 0.5)
                step_revenue += sell_kg * price
                self._held_stock[crop_type] -= sell_kg

        # ---- 6. Water tank update ----
        electricity_cost = water_used_total * 0.5  # INR per liter (pump cost)
        step_cost += electricity_cost

        # Tank refill from rainfall (rooftop collection, ~40% efficiency)
        rainfall_collection = new_weather.rainfall_mm * self.num_slots * 0.4
        new_water = (
            prev_state.resources.water_tank_liters
            - water_used_total
            + rainfall_collection
        )
        new_water = max(0.0, min(self.water_tank_capacity, new_water))

        # Nutrient stock update
        new_nutrient = max(0.0, prev_state.resources.nutrient_stock_units - fertilizer_used_total)
        # Restock cost if low (auto-purchase)
        if new_nutrient < 20:
            restock = 100.0
            step_cost += restock * 8.0  # INR per unit
            new_nutrient += restock

        new_pesticide = max(0.0, prev_state.resources.pesticide_stock_units - pesticide_used_total)
        if new_pesticide < 5 and pesticide_used_total > 0:
            restock = 30.0
            step_cost += restock * 15.0
            new_pesticide += restock

        new_budget = prev_state.resources.budget_inr + step_revenue - step_cost

        new_resources = ResourceState(
            water_tank_liters=round(new_water, 2),
            water_tank_capacity=self.water_tank_capacity,
            nutrient_stock_units=round(new_nutrient, 2),
            pesticide_stock_units=round(new_pesticide, 2),
            electricity_kwh_today=round(water_used_total * 0.002, 3),
            budget_inr=round(new_budget, 2),
            total_revenue_inr=round(prev_state.resources.total_revenue_inr + step_revenue, 2),
            total_cost_inr=round(prev_state.resources.total_cost_inr + step_cost, 2),
        )

        # ---- 7. Eco metrics ----
        new_eco = EcoSimulator.update(
            prev_state.eco,
            water_used_today=water_used_total,
            pesticide_used_today=pesticide_used_total,
            fertilizer_used_today=fertilizer_used_total,
            weather=new_weather,
        )

        # ---- 8. Aggregate metrics ----
        total_yield = prev_state.total_yield_kg + harvested_kg
        total_profit = prev_state.total_profit_inr + (step_revenue - step_cost)
        pest_alert = any(s.pest_pressure > 0.5 for s in updated_slots)

        # ---- 9. Assemble new state ----
        self._state = FarmState(
            day=day,
            episode_length=self.episode_length,
            num_slots=self.num_slots,
            crop_slots=updated_slots,
            weather=new_weather,
            resources=new_resources,
            market=new_market,
            eco=new_eco,
            total_yield_kg=round(total_yield, 3),
            total_profit_inr=round(total_profit, 2),
            active_pest_alert=pest_alert,
        )

        # ---- 10. Reward ----
        reward_breakdown = self.reward_fn.compute(
            prev_state=prev_state,
            action=action,
            next_state=self._state,
            step_revenue_inr=step_revenue,
            step_cost_inr=step_cost,
            harvested_kg=harvested_kg,
            water_used_today_l=water_used_total,
            pesticide_used_today=pesticide_used_total,
        )
        reward = reward_breakdown.total
        self._episode_rewards.append(reward)

        done = day >= self.episode_length

        info = {
            "day": day,
            "step_revenue_inr": round(step_revenue, 2),
            "step_cost_inr": round(step_cost, 2),
            "harvested_kg": round(harvested_kg, 3),
            "water_used_l": round(water_used_total, 2),
            "pesticide_used": round(pesticide_used_total, 2),
            "total_profit_inr": self._state.total_profit_inr,
            "total_yield_kg": self._state.total_yield_kg,
            "eco_score": self._state.eco.eco_score,
            "budget_inr": self._state.resources.budget_inr,
            "reward_breakdown": reward_breakdown.to_dict(),
            "weather": {
                "type": new_weather.weather_type,
                "temp_c": new_weather.temperature_c,
                "rain_mm": new_weather.rainfall_mm,
            },
            "episode_total_reward": sum(self._episode_rewards),
        }

        return self._build_observation(), reward, done, info

    def state(self) -> FarmState:
        """Return the current internal state."""
        assert self._state is not None, "Call reset() before state()"
        return self._state

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> FarmObservation:
        s = self._state
        ep_len = max(s.episode_length, 1)

        # Slot features: [stage/5, health, nutrient_level, water_stress, pest_pressure, days/120]
        slot_features = []
        for slot in s.crop_slots:
            slot_features.extend([
                slot.stage / 5.0,
                slot.health,
                slot.nutrient_level,
                slot.water_stress,
                slot.pest_pressure,
                min(slot.days_since_planted / 120.0, 1.0),
            ])

        forecast = s.weather.forecast_next_3_days
        while len(forecast) < 3:
            forecast.append(0)

        return FarmObservation(
            day_normalized=s.day / ep_len,
            is_monsoon=float(s.weather.is_monsoon_season),
            temperature_norm=min(max((s.weather.temperature_c - 15.0) / 25.0, 0.0), 1.0),
            humidity_norm=s.weather.humidity_pct / 100.0,
            rainfall_norm=min(s.weather.rainfall_mm / 100.0, 1.0),
            weather_type_norm=s.weather.weather_type / 5.0,
            consecutive_dry_norm=min(s.weather.consecutive_dry_days / 30.0, 1.0),
            forecast_day1=forecast[0] / 5.0,
            forecast_day2=forecast[1] / 5.0,
            forecast_day3=forecast[2] / 5.0,
            slot_features=slot_features,
            water_level_norm=s.resources.water_tank_liters / s.resources.water_tank_capacity,
            nutrient_stock_norm=min(s.resources.nutrient_stock_units / 200.0, 1.0),
            pesticide_stock_norm=min(s.resources.pesticide_stock_units / 50.0, 1.0),
            budget_norm=min(max(s.resources.budget_inr / 100000.0, 0.0), 2.0),
            spinach_price_norm=s.market.spinach_price_inr / 35.0,
            lettuce_price_norm=s.market.lettuce_price_inr / 80.0,
            tomato_price_norm=s.market.tomato_price_inr / 25.0,
            herbs_price_norm=s.market.herbs_price_inr / 120.0,
            demand_multiplier=s.market.demand_multiplier,
            eco_score=s.eco.eco_score,
            runoff_risk=s.eco.runoff_risk,
            pest_alert=float(s.active_pest_alert),
        )

    def observation_to_numpy(self, obs: FarmObservation) -> np.ndarray:
        """Convert FarmObservation to flat numpy array for RL agents."""
        return np.array(obs.to_vector(), dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym-compatible wrappers
    # ------------------------------------------------------------------

    def action_space_sample(self) -> FarmAction:
        """Sample a random valid action (for testing / random baselines)."""
        plant = {}
        irrigate = {}
        nutrient = {}
        pest = {}
        harvest = []

        s = self._state
        for slot in s.crop_slots:
            sid = slot.slot_id
            if slot.stage == int(CropStage.EMPTY) and self._rng.random() < 0.3:
                plant[sid] = self._rng.choice(self.VALID_CROPS)
            if slot.stage not in (int(CropStage.EMPTY), int(CropStage.DEAD)):
                irrigate[sid] = round(self._rng.uniform(0.2, 0.8), 2)
                nutrient[sid] = round(self._rng.uniform(0.1, 0.7), 2)
                pest[sid] = self._rng.choice([0, 0, 0, 1, 2])
                if slot.stage == int(CropStage.HARVEST):
                    harvest.append(sid)

        return FarmAction(
            plant_crops=plant,
            irrigation_levels=irrigate,
            nutrient_dose=nutrient,
            pest_control=pest,
            harvest_slots=harvest,
            sell_fraction=round(self._rng.uniform(0.7, 1.0), 2),
        )

    def get_observation_size(self) -> int:
        return self.obs_size

    def render(self) -> str:
        """Simple text render of farm state."""
        if self._state is None:
            return "Environment not initialised."
        s = self._state
        lines = [
            f"=== Day {s.day}/{s.episode_length} ===",
            f"Weather: {WeatherType(s.weather.weather_type).name} | {s.weather.temperature_c}°C | Rain: {s.weather.rainfall_mm}mm",
            f"Budget: ₹{s.resources.budget_inr:,.0f} | Profit: ₹{s.total_profit_inr:,.0f} | Yield: {s.total_yield_kg:.1f}kg",
            f"Water: {s.resources.water_tank_liters:.0f}L | Eco: {s.eco.eco_score:.2f} | Pest Alert: {s.active_pest_alert}",
            "Slots:",
        ]
        for slot in s.crop_slots:
            stage_name = CropStage(slot.stage).name if slot.stage <= 5 else "UNKNOWN"
            lines.append(
                f"  [{slot.slot_id:2d}] {slot.crop_type:8s} | {stage_name:8s} | "
                f"Health:{slot.health:.2f} | Pest:{slot.pest_pressure:.2f} | Yield:{slot.expected_yield_kg:.2f}kg"
            )
        return "\n".join(lines)
