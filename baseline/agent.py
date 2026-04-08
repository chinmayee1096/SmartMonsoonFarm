"""
baseline/agent.py — Expert heuristic baseline agent.

Strategy:
  - Always plant fast-growing, high-value crops when slots are empty
  - Water based on weather (less on rainy days, more on dry/hot days)
  - Dose nutrients on schedule
  - Use biological pest control at low pressure, chemical at high pressure
  - Harvest as soon as crops are mature
  - Hold harvest during low-price days, sell on high-price days
"""

from __future__ import annotations
from typing import Optional
from env.models import FarmState, FarmAction, CropStage, WeatherType
from env.simulator import CROP_CONFIG

VALID_CROPS = ["spinach", "lettuce", "tomato", "herbs"]

# Priority: herbs (high value) > lettuce > spinach (fast) > tomato (slow but high yield)
HEURISTIC_CROP_PRIORITY = ["herbs", "lettuce", "spinach", "tomato"]


class HeuristicAgent:
    """
    Rule-based expert heuristic agent.
    Designed to be a strong baseline — not optimal but sensible.
    """

    def __init__(self, eco_mode: bool = True):
        """
        eco_mode: If True, prefer biological pest control and avoid chemicals.
        """
        self.eco_mode = eco_mode
        self.harvest_hold_days: dict = {}  # slot_id -> days held
        self.price_history: dict = {}

    def act(self, state: FarmState) -> FarmAction:
        plant: dict = {}
        irrigate: dict = {}
        nutrient: dict = {}
        pest: dict = {}
        harvest: list = []

        weather = state.weather
        market = state.market

        # Determine if today is a good selling day
        good_selling_day = market.demand_multiplier >= 1.0 or market.festival_season

        # Base irrigation: 0.5, adjusted for weather
        if weather.weather_type in (int(WeatherType.HEAVY_RAIN),):
            base_irr = 0.0  # rain provides enough water
        elif weather.weather_type in (int(WeatherType.HEATWAVE), int(WeatherType.DRY_SPELL)):
            base_irr = 0.85  # high heat/drought = irrigate more
        elif weather.weather_type == int(WeatherType.LIGHT_RAIN):
            base_irr = 0.2
        elif state.resources.water_tank_liters < 500:
            base_irr = 0.2  # conserve water when tank is low
        else:
            base_irr = 0.5

        for slot in state.crop_slots:
            sid = slot.slot_id

            # --- Planting ---
            if slot.stage == int(CropStage.EMPTY):
                # Choose crop based on slot position and current season
                if state.resources.budget_inr > 5000:  # only plant if we have budget
                    # Monsoon: prefer herbs and spinach (less heat/water sensitive)
                    if weather.is_monsoon_season:
                        crop = "spinach" if sid % 3 == 0 else ("herbs" if sid % 3 == 1 else "lettuce")
                    else:
                        # Dry season: plant tomatoes for higher later yield
                        crop = HEURISTIC_CROP_PRIORITY[sid % len(HEURISTIC_CROP_PRIORITY)]
                    plant[sid] = crop

            elif slot.stage == int(CropStage.DEAD):
                pass  # will auto-clear in environment

            else:
                # --- Irrigation ---
                irrigate[sid] = base_irr

                # Increase irrigation if water stress is high
                if slot.water_stress > 0.5:
                    irrigate[sid] = min(1.0, irrigate[sid] + 0.3)

                # --- Nutrient dosing ---
                # Stage-based nutrition schedule
                if slot.stage == int(CropStage.SEEDING):
                    nutrient[sid] = 0.3  # light feeding for seedlings
                elif slot.stage == int(CropStage.JUVENILE):
                    nutrient[sid] = 0.6  # medium feeding during growth
                elif slot.stage == int(CropStage.MATURE):
                    nutrient[sid] = 0.8  # heavy feeding at maturity
                else:
                    nutrient[sid] = 0.4

                # Reduce if nutrient stock is low
                if state.resources.nutrient_stock_units < 30:
                    nutrient[sid] *= 0.5

                # --- Pest control ---
                if slot.pest_pressure < 0.2:
                    pest[sid] = 0  # no action needed
                elif slot.pest_pressure < 0.5:
                    pest[sid] = 1  # biological control
                else:
                    # High pressure: chemical if not eco_mode, else biological
                    pest[sid] = 1 if self.eco_mode else 2

                # --- Harvest decision ---
                if slot.stage in (int(CropStage.HARVEST),):
                    # Harvest if good selling day OR crop is overdue
                    if good_selling_day or slot.days_since_planted > (
                        CROP_CONFIG.get(slot.crop_type, {}).get("days_to_harvest", 45) + 15
                    ):
                        harvest.append(sid)

                elif slot.stage == int(CropStage.MATURE):
                    # Mature crops: harvest if health is declining
                    if slot.health < 0.7 or slot.pest_pressure > 0.6:
                        harvest.append(sid)

        # Sell fraction: sell more on good days
        sell_fraction = 1.0 if good_selling_day else 0.6

        return FarmAction(
            plant_crops=plant,
            irrigation_levels=irrigate,
            nutrient_dose=nutrient,
            pest_control=pest,
            harvest_slots=harvest,
            sell_fraction=sell_fraction,
        )
