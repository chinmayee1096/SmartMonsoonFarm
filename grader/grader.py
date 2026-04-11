"""
grader/grader.py — Episode graders returning (0, 1) scores ONLY.

Three graders aligned with task difficulty:
  - EasyGrader
  - MediumGrader
  - HardGrader
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from env.models import FarmState


# ✅ NEW: Strict clamp (NO 0.0 or 1.0)
def clamp(x: float) -> float:
    return max(0.01, min(0.99, x))


@dataclass
class GradeResult:
    profit_score: float
    yield_score: float
    eco_score: float
    efficiency_score: float
    composite_score: float
    details: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "profit_score": round(self.profit_score, 4),
            "yield_score": round(self.yield_score, 4),
            "eco_score": round(self.eco_score, 4),
            "efficiency_score": round(self.efficiency_score, 4),
            "composite_score": round(self.composite_score, 4),
            **{k: round(v, 4) for k, v in self.details.items()},
        }


def _sigmoid(x: float, midpoint: float = 0.0, scale: float = 1.0) -> float:
    import math
    try:
        return 1.0 / (1.0 + math.exp(-scale * (x - midpoint)))
    except OverflowError:
        return 1.0 if x > midpoint else 0.0


class EasyGrader:
    TARGET_PROFIT = 20_000.0
    TARGET_YIELD_KG = 30.0
    TARGET_ECO = 0.6

    def grade(self, final_state: FarmState) -> GradeResult:
        profit_raw = final_state.total_profit_inr
        profit_score = _sigmoid(clamp(profit_raw / self.TARGET_PROFIT), midpoint=0.5, scale=5.0)

        yield_raw = final_state.total_yield_kg
        yield_score = clamp(yield_raw / self.TARGET_YIELD_KG)

        eco = final_state.eco.eco_score
        eco_score = clamp((eco - 0.3) / 0.7)

        water_used = final_state.eco.total_water_used_liters
        water_budget = 60 * 8 * 1.0 * 1.2
        efficiency_score = clamp(1.0 - max(0.0, (water_used / water_budget - 1.0)))

        composite = (
            0.45 * profit_score +
            0.30 * yield_score +
            0.15 * eco_score +
            0.10 * efficiency_score
        )

        return GradeResult(
            profit_score=round(clamp(profit_score), 4),
            yield_score=round(clamp(yield_score), 4),
            eco_score=round(clamp(eco_score), 4),
            efficiency_score=round(clamp(efficiency_score), 4),
            composite_score=round(clamp(composite), 4),
            details={
                "total_profit_inr": profit_raw,
                "total_yield_kg": yield_raw,
                "final_eco_score": eco,
                "water_used_l": water_used,
                "budget_remaining_inr": final_state.resources.budget_inr,
            },
        )


class MediumGrader:
    TARGET_PROFIT = 35_000.0
    TARGET_YIELD_KG = 60.0
    TARGET_ECO = 0.7

    def grade(self, final_state: FarmState) -> GradeResult:
        profit_raw = final_state.total_profit_inr
        profit_score = _sigmoid(clamp(profit_raw / self.TARGET_PROFIT), midpoint=0.5, scale=4.0)

        yield_score = clamp(final_state.total_yield_kg / self.TARGET_YIELD_KG)

        eco = final_state.eco.eco_score
        eco_score = clamp((eco - 0.4) / 0.6)

        pest_used = final_state.eco.total_pesticide_used_units
        pest_score = clamp(1.0 - pest_used / 200.0)

        crop_types_grown = set()
        for slot in final_state.crop_slots:
            if slot.crop_type != "none":
                crop_types_grown.add(slot.crop_type)
        diversity_score = clamp(len(crop_types_grown) / 4.0)

        water_used = final_state.eco.total_water_used_liters
        water_budget = 90 * 12 * 1.0 * 1.3
        efficiency_score = clamp(1.0 - max(0.0, water_used / water_budget - 1.0))

        composite = (
            0.35 * profit_score +
            0.25 * yield_score +
            0.20 * eco_score +
            0.10 * pest_score +
            0.05 * diversity_score +
            0.05 * efficiency_score
        )

        return GradeResult(
            profit_score=round(clamp(profit_score), 4),
            yield_score=round(clamp(yield_score), 4),
            eco_score=round(clamp(eco_score), 4),
            efficiency_score=round(clamp(efficiency_score), 4),
            composite_score=round(clamp(composite), 4),
            details={
                "total_profit_inr": profit_raw,
                "total_yield_kg": final_state.total_yield_kg,
                "final_eco_score": eco,
                "pesticide_used": pest_used,
                "crop_diversity": len(crop_types_grown),
                "water_used_l": water_used,
            },
        )


class HardGrader:
    TARGET_PROFIT = 50_000.0
    TARGET_YIELD_KG = 100.0
    TARGET_ECO = 0.8

    def grade(self, final_state: FarmState) -> GradeResult:
        profit_raw = final_state.total_profit_inr
        profit_score = _sigmoid(clamp(profit_raw / self.TARGET_PROFIT), midpoint=0.6, scale=5.0)

        yield_score = clamp(final_state.total_yield_kg / self.TARGET_YIELD_KG)

        eco = final_state.eco.eco_score
        eco_score = clamp((eco - 0.5) / 0.5) if eco > 0.5 else 0.01

        pest_used = final_state.eco.total_pesticide_used_units
        chemical_penalty = clamp(pest_used / 100.0)

        runoff_penalty = clamp(final_state.eco.runoff_risk)

        crop_types_grown = set()
        for slot in final_state.crop_slots:
            if slot.crop_type != "none":
                crop_types_grown.add(slot.crop_type)
        diversity_score = clamp(len(crop_types_grown) / 4.0)

        budget = final_state.resources.budget_inr
        survival_score = 0.99 if budget > 0 else clamp(1.0 + budget / 10000.0)

        water_used = final_state.eco.total_water_used_liters
        water_budget = 120 * 16 * 1.0 * 1.2
        water_efficiency = clamp(1.0 - max(0.0, water_used / water_budget - 1.0))

        composite = (
            0.30 * profit_score +
            0.20 * yield_score +
            0.25 * eco_score +
            0.08 * diversity_score +
            0.07 * survival_score +
            0.05 * water_efficiency +
            -0.05 * chemical_penalty +
            -0.05 * runoff_penalty
        )

        return GradeResult(
            profit_score=round(clamp(profit_score), 4),
            yield_score=round(clamp(yield_score), 4),
            eco_score=round(clamp(eco_score), 4),
            efficiency_score=round(clamp(water_efficiency), 4),
            composite_score=round(clamp(composite), 4),
            details={
                "total_profit_inr": profit_raw,
                "total_yield_kg": final_state.total_yield_kg,
                "final_eco_score": eco,
                "pesticide_used": pest_used,
                "chemical_penalty": chemical_penalty,
                "runoff_penalty": runoff_penalty,
                "crop_diversity": len(crop_types_grown),
                "survival_score": survival_score,
                "water_used_l": water_used,
            },
        )


def get_grader(difficulty: str):
    graders = {
        "easy": EasyGrader(),
        "medium": MediumGrader(),
        "hard": HardGrader(),
    }
    if difficulty not in graders:
        raise ValueError(f"Unknown difficulty: {difficulty}. Choose from {list(graders)}")
    return graders[difficulty]
