"""
Pydantic models for the Smart Monsoon-Resilient Urban Farm environment.
Defines typed State, Action, and Observation schemas.
"""

from __future__ import annotations
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator
from enum import IntEnum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CropStage(IntEnum):
    EMPTY    = 0   # No crop planted
    SEEDING  = 1   # Days 0-7 after planting
    JUVENILE = 2   # Days 8-21
    MATURE   = 3   # Days 22-40
    HARVEST  = 4   # Days 41+  (ready to harvest)
    DEAD     = 5   # Crop lost to drought / pest / disease


class WeatherType(IntEnum):
    SUNNY     = 0
    CLOUDY    = 1
    LIGHT_RAIN = 2
    HEAVY_RAIN = 3
    HEATWAVE  = 4
    DRY_SPELL = 5


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class CropSlot(BaseModel):
    """Represents a single growing slot / tray in the hydroponic system."""
    slot_id: int
    crop_type: str = "none"           # "spinach", "lettuce", "tomato", "herbs", "none"
    stage: CropStage = CropStage.EMPTY
    days_since_planted: int = 0
    health: float = Field(default=1.0, ge=0.0, le=1.0)  # 0=dead, 1=perfect
    nutrient_level: float = Field(default=0.5, ge=0.0, le=1.0)
    water_stress: float = Field(default=0.0, ge=0.0, le=1.0)  # 0=none, 1=critical
    pest_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_yield_kg: float = 0.0    # Projected yield at harvest


class WeatherState(BaseModel):
    """Current and forecast weather conditions (Bengaluru climate)."""
    weather_type: WeatherType = WeatherType.SUNNY
    temperature_c: float = 28.0       # Celsius
    humidity_pct: float = 60.0        # Percent
    rainfall_mm: float = 0.0          # Daily rainfall
    solar_radiation: float = 0.8      # 0-1 fraction of max
    is_monsoon_season: bool = False    # True: June-September
    consecutive_dry_days: int = 0
    forecast_next_3_days: List[int] = Field(default_factory=lambda: [0, 0, 0])


class ResourceState(BaseModel):
    """Water tank, nutrients, electricity and budget tracking."""
    water_tank_liters: float = Field(default=5000.0, ge=0.0)
    water_tank_capacity: float = 5000.0
    nutrient_stock_units: float = Field(default=200.0, ge=0.0)   # abstract units
    pesticide_stock_units: float = Field(default=50.0, ge=0.0)
    electricity_kwh_today: float = 0.0
    budget_inr: float = 50000.0       # Starting budget in Indian Rupees
    total_revenue_inr: float = 0.0
    total_cost_inr: float = 0.0


class MarketState(BaseModel):
    """Simulated mandi (wholesale market) prices in INR/kg."""
    spinach_price_inr: float = 35.0
    lettuce_price_inr: float = 80.0
    tomato_price_inr: float = 25.0
    herbs_price_inr: float = 120.0
    demand_multiplier: float = 1.0    # 0.5 (glut) to 2.0 (shortage)
    festival_season: bool = False     # Higher demand


class EcoMetrics(BaseModel):
    """Sustainability tracking."""
    total_water_used_liters: float = 0.0
    total_pesticide_used_units: float = 0.0
    total_fertilizer_used_units: float = 0.0
    runoff_risk: float = 0.0          # 0-1, high during heavy rain + chemical use
    eco_score: float = 1.0            # 1.0 = perfect, degrades with overuse


# ---------------------------------------------------------------------------
# Top-level State
# ---------------------------------------------------------------------------

class FarmState(BaseModel):
    """Complete observable state of the hydroponic farm."""
    # Time
    day: int = 0
    episode_length: int = 90
    
    # Farm layout
    num_slots: int = 12
    crop_slots: List[CropSlot] = Field(default_factory=list)
    
    # Environment
    weather: WeatherState = Field(default_factory=WeatherState)
    
    # Resources
    resources: ResourceState = Field(default_factory=ResourceState)
    
    # Market
    market: MarketState = Field(default_factory=MarketState)
    
    # Eco
    eco: EcoMetrics = Field(default_factory=EcoMetrics)
    
    # Aggregates (updated each step)
    total_yield_kg: float = 0.0
    total_profit_inr: float = 0.0
    active_pest_alert: bool = False
    
    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------

class FarmAction(BaseModel):
    """
    Composite action for one day of farm management.
    All values are per-slot or farm-wide continuous/discrete controls.
    """
    # Planting decisions: list of (slot_id, crop_type) or empty
    plant_crops: Dict[int, str] = Field(
        default_factory=dict,
        description="slot_id -> crop_type to plant. Use 'none' to skip."
    )
    
    # Irrigation: per-slot water amount (0.0 = no water, 1.0 = maximum dose)
    irrigation_levels: Dict[int, float] = Field(
        default_factory=dict,
        description="slot_id -> irrigation fraction [0.0, 1.0]"
    )
    
    # Nutrient dosing: per-slot (0.0 = none, 1.0 = max dose)
    nutrient_dose: Dict[int, float] = Field(
        default_factory=dict,
        description="slot_id -> nutrient fraction [0.0, 1.0]"
    )
    
    # Pest control: per-slot (0=none, 1=biological, 2=chemical)
    pest_control: Dict[int, int] = Field(
        default_factory=dict,
        description="slot_id -> pest control type: 0=none,1=biological,2=chemical"
    )
    
    # Harvest decisions: list of slot_ids to harvest
    harvest_slots: List[int] = Field(
        default_factory=list,
        description="Slot IDs to harvest today"
    )
    
    # Sales decisions: how much of harvested stock to sell vs store
    sell_fraction: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Fraction of today's harvest to sell (rest held for better price)"
    )
    
    @validator('irrigation_levels', 'nutrient_dose', each_item=False)
    def clamp_fractions(cls, v):
        return {k: max(0.0, min(1.0, val)) for k, val in v.items()}
    
    @validator('pest_control', each_item=False)
    def validate_pest_control(cls, v):
        return {k: max(0, min(2, val)) for k, val in v.items()}


# ---------------------------------------------------------------------------
# Observation Model (flattened for RL agents)
# ---------------------------------------------------------------------------

class FarmObservation(BaseModel):
    """
    Flattened observation vector for RL agents.
    Normalized to [0, 1] where applicable.
    """
    # Time
    day_normalized: float             # day / episode_length
    is_monsoon: float                 # 0 or 1
    
    # Weather (normalized)
    temperature_norm: float           # (temp - 15) / 25
    humidity_norm: float              # humidity / 100
    rainfall_norm: float              # min(rainfall / 100, 1.0)
    weather_type_norm: float          # weather_type / 5
    consecutive_dry_norm: float       # min(dry_days / 30, 1.0)
    forecast_day1: float
    forecast_day2: float
    forecast_day3: float
    
    # Per-slot observations (flattened, 12 slots * 6 features = 72)
    slot_features: List[float]        # [stage, health, nutrient, water_stress, pest, days_norm] * n_slots
    
    # Resources (normalized)
    water_level_norm: float           # tank / capacity
    nutrient_stock_norm: float        # stock / 200
    pesticide_stock_norm: float       # stock / 50
    budget_norm: float                # budget / 100000
    
    # Market prices (normalized relative to baseline)
    spinach_price_norm: float
    lettuce_price_norm: float
    tomato_price_norm: float
    herbs_price_norm: float
    demand_multiplier: float
    
    # Eco metrics
    eco_score: float
    runoff_risk: float
    
    # Alert flags
    pest_alert: float                 # 0 or 1
    
    def to_vector(self) -> List[float]:
        """Return flat numpy-ready list."""
        base = [
            self.day_normalized, self.is_monsoon,
            self.temperature_norm, self.humidity_norm,
            self.rainfall_norm, self.weather_type_norm,
            self.consecutive_dry_norm,
            self.forecast_day1, self.forecast_day2, self.forecast_day3,
            self.water_level_norm, self.nutrient_stock_norm,
            self.pesticide_stock_norm, self.budget_norm,
            self.spinach_price_norm, self.lettuce_price_norm,
            self.tomato_price_norm, self.herbs_price_norm,
            self.demand_multiplier,
            self.eco_score, self.runoff_risk,
            self.pest_alert,
        ]
        return base + self.slot_features
