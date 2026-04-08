"""
simulator.py — Physics and stochastic dynamics engine for the farm.

Implements:
  - Bengaluru climate model (monthly rainfall normals, temperature ranges)
  - Crop growth model (stage transitions, health, yield)
  - Pest outbreak model (probabilistic, weather-correlated)
  - Market price fluctuation (random walk with seasonality)
  - Resource consumption calculations
"""

from __future__ import annotations
import math
import random
from typing import Tuple, List, Dict

from env.models import (
    FarmState, CropSlot, WeatherState, ResourceState,
    MarketState, EcoMetrics, WeatherType, CropStage
)


# ---------------------------------------------------------------------------
# Bengaluru Climate Constants (monthly averages)
# ---------------------------------------------------------------------------

# (mean_temp_C, mean_rainfall_mm/day, humidity_pct, solar_fraction)
BENGALURU_MONTHLY_CLIMATE = {
    1:  (21.5, 0.2,  55, 0.85),   # January
    2:  (23.0, 0.3,  50, 0.88),   # February
    3:  (26.0, 0.5,  48, 0.85),   # March
    4:  (28.5, 3.5,  55, 0.75),   # April  (pre-monsoon)
    5:  (27.5, 8.0,  65, 0.65),   # May    (pre-monsoon)
    6:  (24.0, 14.0, 80, 0.45),   # June   (SW Monsoon onset)
    7:  (23.0, 12.0, 85, 0.40),   # July
    8:  (23.5, 13.0, 83, 0.42),   # August
    9:  (24.0, 15.0, 80, 0.48),   # September
    10: (24.5, 12.0, 75, 0.55),   # October (NE Monsoon)
    11: (22.5, 4.0,  68, 0.70),   # November
    12: (20.5, 0.8,  60, 0.80),   # December
}

# Monsoon months for Bengaluru
MONSOON_MONTHS = {6, 7, 8, 9, 10}

# Crop configuration
CROP_CONFIG = {
    "spinach": {
        "days_to_harvest": 35,
        "water_per_day_l": 0.8,      # liters per slot per day
        "nutrient_per_day": 0.4,
        "base_yield_kg": 0.6,
        "price_base_inr": 35.0,
        "heat_tolerance": 0.4,        # 0=sensitive, 1=tolerant
        "drought_tolerance": 0.3,
        "seed_cost_inr": 50,
    },
    "lettuce": {
        "days_to_harvest": 45,
        "water_per_day_l": 1.0,
        "nutrient_per_day": 0.5,
        "base_yield_kg": 0.8,
        "price_base_inr": 80.0,
        "heat_tolerance": 0.3,
        "drought_tolerance": 0.25,
        "seed_cost_inr": 80,
    },
    "tomato": {
        "days_to_harvest": 75,
        "water_per_day_l": 1.5,
        "nutrient_per_day": 0.8,
        "base_yield_kg": 2.5,
        "price_base_inr": 25.0,
        "heat_tolerance": 0.6,
        "drought_tolerance": 0.5,
        "seed_cost_inr": 40,
    },
    "herbs": {
        "days_to_harvest": 30,
        "water_per_day_l": 0.5,
        "nutrient_per_day": 0.3,
        "base_yield_kg": 0.3,
        "price_base_inr": 120.0,
        "heat_tolerance": 0.5,
        "drought_tolerance": 0.55,
        "seed_cost_inr": 100,
    },
}


class WeatherSimulator:
    """Stochastic weather model calibrated to Bengaluru climate."""

    def __init__(self, start_day: int = 1, seed: int = 42):
        self.rng = random.Random(seed)
        self.start_day = start_day  # day-of-year (1-365)

    def _month_from_day(self, episode_day: int) -> int:
        doy = (self.start_day + episode_day - 1) % 365 + 1
        # approximate month from day-of-year
        breakpoints = [0,31,59,90,120,151,181,212,243,273,304,334,365]
        for m, (lo, hi) in enumerate(zip(breakpoints, breakpoints[1:]), 1):
            if lo < doy <= hi:
                return m
        return 12

    def simulate_day(self, prev_state: WeatherState, episode_day: int) -> WeatherState:
        month = self._month_from_day(episode_day)
        mean_temp, mean_rain, mean_humidity, solar = BENGALURU_MONTHLY_CLIMATE[month]
        is_monsoon = month in MONSOON_MONTHS

        # Temperature: normal distribution around monthly mean
        temp_std = 3.0 if not is_monsoon else 2.0
        temperature = self.rng.gauss(mean_temp, temp_std)
        temperature = max(15.0, min(42.0, temperature))

        # Heatwave: May-June, <5% chance per day
        is_heatwave = (month in {4, 5, 6}) and self.rng.random() < 0.05
        if is_heatwave:
            temperature = min(42.0, temperature + self.rng.uniform(4, 8))

        # Humidity
        humidity = self.rng.gauss(mean_humidity, 10.0)
        humidity = max(20.0, min(100.0, humidity))

        # Rainfall
        rain_prob = min(0.9, mean_rain / 20.0)  # probability of rain today
        if self.rng.random() < rain_prob:
            # Exponential rainfall amount
            rainfall = self.rng.expovariate(1.0 / mean_rain)
            rainfall = min(rainfall, 200.0)  # cap extreme events
            # Heavy rain chance during monsoon
            if is_monsoon and self.rng.random() < 0.15:
                rainfall = min(200.0, rainfall * self.rng.uniform(2, 4))
        else:
            rainfall = 0.0

        # Determine weather type
        if is_heatwave:
            wtype = WeatherType.HEATWAVE
        elif rainfall > 30:
            wtype = WeatherType.HEAVY_RAIN
        elif rainfall > 5:
            wtype = WeatherType.LIGHT_RAIN
        elif humidity > 80:
            wtype = WeatherType.CLOUDY
        else:
            wtype = WeatherType.SUNNY

        # Dry spell tracking
        consecutive_dry = (
            prev_state.consecutive_dry_days + 1
            if rainfall < 1.0
            else 0
        )
        if consecutive_dry > 10 and not is_monsoon:
            wtype = WeatherType.DRY_SPELL

        # Solar radiation (lower on rainy/cloudy days)
        solar_actual = solar * self.rng.uniform(0.7, 1.3)
        if rainfall > 10:
            solar_actual *= 0.4
        elif rainfall > 2:
            solar_actual *= 0.7
        solar_actual = max(0.1, min(1.0, solar_actual))

        # 3-day forecast (simplified)
        forecast = []
        for i in range(1, 4):
            fm = self._month_from_day(episode_day + i)
            _, fr, _, _ = BENGALURU_MONTHLY_CLIMATE[fm]
            fp = min(0.9, fr / 20.0)
            if self.rng.random() < fp:
                forecast.append(int(WeatherType.LIGHT_RAIN))
            else:
                forecast.append(int(WeatherType.SUNNY))

        return WeatherState(
            weather_type=int(wtype),
            temperature_c=round(temperature, 1),
            humidity_pct=round(humidity, 1),
            rainfall_mm=round(rainfall, 1),
            solar_radiation=round(solar_actual, 3),
            is_monsoon_season=is_monsoon,
            consecutive_dry_days=consecutive_dry,
            forecast_next_3_days=forecast,
        )


class CropSimulator:
    """Models hydroponic crop growth, health dynamics and yield estimation."""

    @staticmethod
    def stage_for_days(crop_type: str, days: int) -> CropStage:
        if crop_type == "none" or days < 0:
            return CropStage.EMPTY
        cfg = CROP_CONFIG.get(crop_type, {})
        dth = cfg.get("days_to_harvest", 45)
        if days == 0:
            return CropStage.SEEDING
        elif days <= 7:
            return CropStage.SEEDING
        elif days <= int(dth * 0.5):
            return CropStage.JUVENILE
        elif days < dth:
            return CropStage.MATURE
        else:
            return CropStage.HARVEST

    @staticmethod
    def compute_water_stress(
        slot: CropSlot,
        water_given_l: float,
        weather: WeatherState,
    ) -> float:
        """Compute water stress after irrigation. Returns new water_stress [0,1]."""
        cfg = CROP_CONFIG.get(slot.crop_type, {})
        water_needed = cfg.get("water_per_day_l", 1.0)

        # Rainfall contribution (rooftop hydroponic: partial rainwater collection)
        rain_contribution = min(weather.rainfall_mm * 0.05, water_needed * 0.6)
        effective_water = water_given_l + rain_contribution

        ratio = effective_water / max(water_needed, 0.1)

        if ratio >= 0.9:
            return max(0.0, slot.water_stress - 0.15)
        elif ratio >= 0.6:
            return max(0.0, slot.water_stress - 0.05)
        else:
            # Underwatered — stress increases
            deficit_penalty = (0.6 - ratio) * 0.3
            # Heatwave makes it worse
            if weather.weather_type == int(WeatherType.HEATWAVE):
                deficit_penalty *= 1.5
            return min(1.0, slot.water_stress + deficit_penalty)

    @staticmethod
    def compute_health(
        slot: CropSlot,
        new_water_stress: float,
        nutrient_given: float,
        weather: WeatherState,
    ) -> float:
        """Update crop health based on stressors."""
        cfg = CROP_CONFIG.get(slot.crop_type, {})
        health = slot.health

        # Nutrient effect
        nutrient_needed = cfg.get("nutrient_per_day", 0.5)
        nutrient_ratio = nutrient_given / max(nutrient_needed, 0.01)
        if nutrient_ratio >= 0.8:
            health += 0.01  # thriving
        elif nutrient_ratio < 0.3:
            health -= 0.03  # deficient
        if nutrient_ratio > 1.5:
            health -= 0.02  # over-fertilised

        # Water stress effect
        if new_water_stress > 0.7:
            health -= 0.05
        elif new_water_stress > 0.4:
            health -= 0.02

        # Pest pressure
        if slot.pest_pressure > 0.6:
            health -= 0.08
        elif slot.pest_pressure > 0.3:
            health -= 0.03

        # Heat stress (heat-sensitive crops in heatwave)
        if weather.weather_type == int(WeatherType.HEATWAVE):
            heat_tolerance = cfg.get("heat_tolerance", 0.5)
            health -= (1.0 - heat_tolerance) * 0.05

        # Solar boost during growth stages
        if slot.stage in (int(CropStage.JUVENILE), int(CropStage.MATURE)):
            health += weather.solar_radiation * 0.005

        return max(0.0, min(1.0, health))

    @staticmethod
    def compute_expected_yield(slot: CropSlot) -> float:
        """Estimate kg yield at harvest given current health and stage."""
        cfg = CROP_CONFIG.get(slot.crop_type, {})
        base = cfg.get("base_yield_kg", 1.0)
        return round(base * slot.health * (1.0 - slot.water_stress * 0.5), 3)

    @staticmethod
    def update_pest_pressure(
        slot: CropSlot,
        weather: WeatherState,
        pest_control_action: int,
        rng: random.Random,
    ) -> float:
        """Stochastic pest pressure update."""
        pressure = slot.pest_pressure

        # Base spread probability
        base_prob = 0.08
        # Warm + humid = pest heaven
        if weather.humidity_pct > 75 and weather.temperature_c > 25:
            base_prob *= 1.6
        if weather.weather_type == int(WeatherType.CLOUDY):
            base_prob *= 1.3

        # Natural spread
        if rng.random() < base_prob:
            pressure = min(1.0, pressure + rng.uniform(0.05, 0.15))

        # Control actions
        if pest_control_action == 1:  # biological
            pressure = max(0.0, pressure - rng.uniform(0.10, 0.20))
        elif pest_control_action == 2:  # chemical
            pressure = max(0.0, pressure - rng.uniform(0.25, 0.40))

        # Natural recovery if low
        if pressure < 0.1 and rng.random() < 0.2:
            pressure = max(0.0, pressure - 0.02)

        return round(pressure, 3)


class MarketSimulator:
    """Simulated mandi price dynamics with seasonality and random walk."""

    BASE_PRICES = {
        "spinach": 35.0,
        "lettuce": 80.0,
        "tomato": 25.0,
        "herbs":   120.0,
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed + 1)

    def step(self, prev_market: MarketState, day: int) -> MarketState:
        # Demand multiplier: festival boost (e.g., Diwali around day 50-60 if start=Aug)
        festival = (40 <= day <= 55)
        demand = prev_market.demand_multiplier
        # Mean-reverting random walk
        demand += self.rng.gauss(0, 0.05)
        demand = max(0.5, min(2.0, demand))
        if festival:
            demand = min(2.0, demand + 0.1)

        # Price random walk for each crop
        def update_price(prev_price: float, base: float) -> float:
            noise = self.rng.gauss(0, base * 0.04)
            mean_revert = (base - prev_price) * 0.1
            new_price = prev_price + noise + mean_revert
            return round(max(base * 0.4, min(base * 2.5, new_price)), 2)

        return MarketState(
            spinach_price_inr=update_price(prev_market.spinach_price_inr, self.BASE_PRICES["spinach"]),
            lettuce_price_inr=update_price(prev_market.lettuce_price_inr, self.BASE_PRICES["lettuce"]),
            tomato_price_inr=update_price(prev_market.tomato_price_inr, self.BASE_PRICES["tomato"]),
            herbs_price_inr=update_price(prev_market.herbs_price_inr, self.BASE_PRICES["herbs"]),
            demand_multiplier=round(demand, 3),
            festival_season=festival,
        )

    def get_price(self, market: MarketState, crop_type: str) -> float:
        mapping = {
            "spinach": market.spinach_price_inr,
            "lettuce": market.lettuce_price_inr,
            "tomato":  market.tomato_price_inr,
            "herbs":   market.herbs_price_inr,
        }
        base_price = mapping.get(crop_type, 0.0)
        return round(base_price * market.demand_multiplier, 2)


class EcoSimulator:
    """Tracks environmental impact and computes eco-score."""

    @staticmethod
    def update(
        eco: EcoMetrics,
        water_used_today: float,
        pesticide_used_today: float,
        fertilizer_used_today: float,
        weather: WeatherState,
    ) -> EcoMetrics:
        total_water   = eco.total_water_used_liters + water_used_today
        total_pest    = eco.total_pesticide_used_units + pesticide_used_today
        total_fert    = eco.total_fertilizer_used_units + fertilizer_used_today

        # Runoff risk: high rainfall + chemical use = bad
        runoff = 0.0
        if weather.rainfall_mm > 20 and (pesticide_used_today + fertilizer_used_today) > 0:
            runoff = min(1.0, (pesticide_used_today + fertilizer_used_today) * 0.1)
        else:
            runoff = max(0.0, eco.runoff_risk - 0.05)

        # Eco score: penalise chemical overuse and runoff
        eco_score = eco.eco_score
        if pesticide_used_today > 5:
            eco_score -= 0.02
        if fertilizer_used_today > 10:
            eco_score -= 0.01
        if runoff > 0.5:
            eco_score -= 0.03
        # Slight recovery if clean day
        if pesticide_used_today == 0 and runoff < 0.1:
            eco_score = min(1.0, eco_score + 0.002)

        eco_score = max(0.0, min(1.0, eco_score))

        return EcoMetrics(
            total_water_used_liters=round(total_water, 2),
            total_pesticide_used_units=round(total_pest, 2),
            total_fertilizer_used_units=round(total_fert, 2),
            runoff_risk=round(runoff, 3),
            eco_score=round(eco_score, 4),
        )
