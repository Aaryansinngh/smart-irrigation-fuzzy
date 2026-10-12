"""
Mamdani Fuzzy Inference System for Smart Irrigation Recommendation.

Tailored for Pearl Millet (Bajra) cultivation in Jaipur District, Rajasthan.
Translates soil moisture, ambient temperature, relative humidity, and recent rainfall
into an interpretable, defuzzified daily irrigation recommendation (mm/day).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


@dataclass
class RuleActivation:
    rule_id: int
    rule_text: str
    strength: float  # alpha-cut activation level in [0, 1]
    consequent: str  # linguistic category (None, Low, Medium, High)


@dataclass
class RecommendationResult:
    irrigation_depth_mm: float
    category: str
    linguistic_summary: str
    soil_moisture_pct: float
    temperature_c: float
    relative_humidity_pct: float
    recent_rainfall_mm: float
    crop_stage: str
    membership_degrees: Dict[str, Dict[str, float]]
    active_rules: List[RuleActivation]
    aggregated_fuzzy_curve: Tuple[np.ndarray, np.ndarray]
    warnings: List[str] = field(default_factory=list)


class PearlMilletIrrigationFIS:
    """Mamdani Fuzzy Inference System for Pearl Millet irrigation scheduling."""

    def __init__(self):
        # 1. Define Universes of Discourse
        # Soil Moisture: 0% to 100% root-zone wetness
        self.x_sm = np.linspace(0, 100, 501)
        # Temperature: 15°C to 48°C daily mean
        self.x_temp = np.linspace(15, 48, 331)
        # Relative Humidity: 10% to 100% daily mean
        self.x_rh = np.linspace(10, 100, 451)
        # Recent Rainfall: 0 mm to 60 mm 3-day cumulative
        self.x_rain = np.linspace(0, 60, 301)
        # Irrigation Output: 0 mm to 12 mm/day net irrigation requirement
        self.x_irr = np.linspace(0, 12, 601)

        # 2. Define Membership Functions (scikit-fuzzy)
        # Soil Moisture: Dry, Moderate, Wet
        self.sm_dry = fuzz.trapmf(self.x_sm, [0, 0, 20, 38])
        self.sm_moderate = fuzz.trimf(self.x_sm, [28, 48, 68])
        self.sm_wet = fuzz.trapmf(self.x_sm, [58, 75, 100, 100])

        # Temperature: Cool, Moderate, Hot
        self.temp_cool = fuzz.trapmf(self.x_temp, [15, 15, 22, 28])
        self.temp_moderate = fuzz.trimf(self.x_temp, [25, 32, 38])
        self.temp_hot = fuzz.trapmf(self.x_temp, [35, 40, 48, 48])

        # Relative Humidity: Low, Moderate, High
        self.rh_low = fuzz.trapmf(self.x_rh, [10, 10, 25, 45])
        self.rh_moderate = fuzz.trimf(self.x_rh, [35, 55, 75])
        self.rh_high = fuzz.trapmf(self.x_rh, [65, 80, 100, 100])

        # Recent Rainfall: Low, Moderate, High
        self.rain_low = fuzz.trapmf(self.x_rain, [0, 0, 5, 12])
        self.rain_moderate = fuzz.trimf(self.x_rain, [8, 20, 35])
        self.rain_high = fuzz.trapmf(self.x_rain, [28, 45, 60, 60])

        # Output: Irrigation Requirement (mm/day)
        self.irr_none = fuzz.trapmf(self.x_irr, [0, 0, 0.5, 2.0])
        self.irr_low = fuzz.trimf(self.x_irr, [1.0, 3.0, 5.0])
        self.irr_medium = fuzz.trimf(self.x_irr, [4.0, 6.5, 9.0])
        self.irr_high = fuzz.trapmf(self.x_irr, [7.5, 9.5, 12.0, 12.0])

        # Initialize skfuzzy control system
        self._build_control_system()

    def _build_control_system(self):
        """Builds scikit-fuzzy ControlSystem as alternative execution engine."""
        sm_var = ctrl.Antecedent(self.x_sm, "soil_moisture")
        temp_var = ctrl.Antecedent(self.x_temp, "temperature")
        rh_var = ctrl.Antecedent(self.x_rh, "relative_humidity")
        rain_var = ctrl.Antecedent(self.x_rain, "recent_rainfall")
        irr_var = ctrl.Consequent(self.x_irr, "irrigation")

        sm_var["dry"] = self.sm_dry
        sm_var["moderate"] = self.sm_moderate
        sm_var["wet"] = self.sm_wet

        temp_var["cool"] = self.temp_cool
        temp_var["moderate"] = self.temp_moderate
        temp_var["hot"] = self.temp_hot

        rh_var["low"] = self.rh_low
        rh_var["moderate"] = self.rh_moderate
        rh_var["high"] = self.rh_high

        rain_var["low"] = self.rain_low
        rain_var["moderate"] = self.rain_moderate
        rain_var["high"] = self.rain_high

        irr_var["none"] = self.irr_none
        irr_var["low"] = self.irr_low
        irr_var["medium"] = self.irr_medium
        irr_var["high"] = self.irr_high

        # Rules
        rules = [
            # Wet soil / heavy rain conditions
            ctrl.Rule(sm_var["wet"] | rain_var["high"], irr_var["none"]),
            ctrl.Rule(sm_var["wet"] & rain_var["moderate"], irr_var["none"]),
            ctrl.Rule(sm_var["wet"] & rain_var["low"] & temp_var["cool"], irr_var["none"]),
            ctrl.Rule(sm_var["wet"] & rain_var["low"] & temp_var["hot"], irr_var["low"]),
            # Moderate soil conditions
            ctrl.Rule(sm_var["moderate"] & rain_var["high"], irr_var["none"]),
            ctrl.Rule(sm_var["moderate"] & rain_var["moderate"] & rh_var["high"], irr_var["none"]),
            ctrl.Rule(sm_var["moderate"] & rain_var["moderate"] & rh_var["moderate"], irr_var["low"]),
            ctrl.Rule(sm_var["moderate"] & rain_var["low"] & temp_var["cool"], irr_var["low"]),
            ctrl.Rule(sm_var["moderate"] & rain_var["low"] & temp_var["moderate"], irr_var["medium"]),
            ctrl.Rule(sm_var["moderate"] & rain_var["low"] & temp_var["hot"] & rh_var["low"], irr_var["high"]),
            ctrl.Rule(sm_var["moderate"] & rain_var["low"] & temp_var["hot"] & rh_var["moderate"], irr_var["medium"]),
            # Dry soil conditions
            ctrl.Rule(sm_var["dry"] & rain_var["high"], irr_var["low"]),
            ctrl.Rule(sm_var["dry"] & rain_var["moderate"], irr_var["medium"]),
            ctrl.Rule(sm_var["dry"] & rain_var["low"] & temp_var["cool"], irr_var["medium"]),
            ctrl.Rule(sm_var["dry"] & rain_var["low"] & temp_var["moderate"], irr_var["high"]),
            ctrl.Rule(sm_var["dry"] & rain_var["low"] & temp_var["hot"], irr_var["high"]),
            ctrl.Rule(sm_var["dry"] & rh_var["low"], irr_var["high"]),
        ]

        self.control_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    def calculate_membership_degrees(
        self,
        soil_moisture: float,
        temperature: float,
        relative_humidity: float,
        recent_rainfall: float,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate exact scalar membership degree for each input linguistic set."""
        sm_c = np.clip(soil_moisture, 0, 100)
        temp_c = np.clip(temperature, 15, 48)
        rh_c = np.clip(relative_humidity, 10, 100)
        rain_c = np.clip(recent_rainfall, 0, 60)

        memberships = {
            "soil_moisture": {
                "dry": float(fuzz.interp_membership(self.x_sm, self.sm_dry, sm_c)),
                "moderate": float(fuzz.interp_membership(self.x_sm, self.sm_moderate, sm_c)),
                "wet": float(fuzz.interp_membership(self.x_sm, self.sm_wet, sm_c)),
            },
            "temperature": {
                "cool": float(fuzz.interp_membership(self.x_temp, self.temp_cool, temp_c)),
                "moderate": float(fuzz.interp_membership(self.x_temp, self.temp_moderate, temp_c)),
                "hot": float(fuzz.interp_membership(self.x_temp, self.temp_hot, temp_c)),
            },
            "relative_humidity": {
                "low": float(fuzz.interp_membership(self.x_rh, self.rh_low, rh_c)),
                "moderate": float(fuzz.interp_membership(self.x_rh, self.rh_moderate, rh_c)),
                "high": float(fuzz.interp_membership(self.x_rh, self.rh_high, rh_c)),
            },
            "recent_rainfall": {
                "low": float(fuzz.interp_membership(self.x_rain, self.rain_low, rain_c)),
                "moderate": float(fuzz.interp_membership(self.x_rain, self.rain_moderate, rain_c)),
                "high": float(fuzz.interp_membership(self.x_rain, self.rain_high, rain_c)),
            },
        }
        return memberships

    def evaluate_rules_and_aggregate(
        self,
        memberships: Dict[str, Dict[str, float]],
    ) -> Tuple[List[RuleActivation], np.ndarray]:
        """Evaluates 17 agronomic Mamdani rules and computes max-min aggregated output curve."""
        sm = memberships["soil_moisture"]
        temp = memberships["temperature"]
        rh = memberships["relative_humidity"]
        rain = memberships["recent_rainfall"]

        rule_definitions = [
            (1, "IF Soil is Wet OR Recent Rain is High THEN Irrigation is None",
             max(sm["wet"], rain["high"]), "none", self.irr_none),
            (2, "IF Soil is Wet AND Recent Rain is Moderate THEN Irrigation is None",
             min(sm["wet"], rain["moderate"]), "none", self.irr_none),
            (3, "IF Soil is Wet AND Recent Rain is Low AND Temp is Cool THEN Irrigation is None",
             min(sm["wet"], rain["low"], temp["cool"]), "none", self.irr_none),
            (4, "IF Soil is Wet AND Recent Rain is Low AND Temp is Hot THEN Irrigation is Low",
             min(sm["wet"], rain["low"], temp["hot"]), "low", self.irr_low),
            (5, "IF Soil is Moderate AND Recent Rain is High THEN Irrigation is None",
             min(sm["moderate"], rain["high"]), "none", self.irr_none),
            (6, "IF Soil is Moderate AND Recent Rain is Moderate AND RH is High THEN Irrigation is None",
             min(sm["moderate"], rain["moderate"], rh["high"]), "none", self.irr_none),
            (7, "IF Soil is Moderate AND Recent Rain is Moderate AND RH is Moderate THEN Irrigation is Low",
             min(sm["moderate"], rain["moderate"], rh["moderate"]), "low", self.irr_low),
            (8, "IF Soil is Moderate AND Recent Rain is Low AND Temp is Cool THEN Irrigation is Low",
             min(sm["moderate"], rain["low"], temp["cool"]), "low", self.irr_low),
            (9, "IF Soil is Moderate AND Recent Rain is Low AND Temp is Moderate THEN Irrigation is Medium",
             min(sm["moderate"], rain["low"], temp["moderate"]), "medium", self.irr_medium),
            (10, "IF Soil is Moderate AND Recent Rain is Low AND Temp is Hot AND RH is Low THEN Irrigation is High",
             min(sm["moderate"], rain["low"], temp["hot"], rh["low"]), "high", self.irr_high),
            (11, "IF Soil is Moderate AND Recent Rain is Low AND Temp is Hot AND RH is Moderate THEN Irrigation is Medium",
             min(sm["moderate"], rain["low"], temp["hot"], rh["moderate"]), "medium", self.irr_medium),
            (12, "IF Soil is Dry AND Recent Rain is High THEN Irrigation is Low",
             min(sm["dry"], rain["high"]), "low", self.irr_low),
            (13, "IF Soil is Dry AND Recent Rain is Moderate THEN Irrigation is Medium",
             min(sm["dry"], rain["moderate"]), "medium", self.irr_medium),
            (14, "IF Soil is Dry AND Recent Rain is Low AND Temp is Cool THEN Irrigation is Medium",
             min(sm["dry"], rain["low"], temp["cool"]), "medium", self.irr_medium),
            (15, "IF Soil is Dry AND Recent Rain is Low AND Temp is Moderate THEN Irrigation is High",
             min(sm["dry"], rain["low"], temp["moderate"]), "high", self.irr_high),
            (16, "IF Soil is Dry AND Recent Rain is Low AND Temp is Hot THEN Irrigation is High",
             min(sm["dry"], rain["low"], temp["hot"]), "high", self.irr_high),
            (17, "IF Soil is Dry AND RH is Low THEN Irrigation is High",
             min(sm["dry"], rh["low"]), "high", self.irr_high),
        ]

        active_rules: List[RuleActivation] = []
        aggregated = np.zeros_like(self.x_irr)

        for rule_id, text, alpha, consequent_label, mf_curve in rule_definitions:
            if alpha > 0.001:
                active_rules.append(
                    RuleActivation(
                        rule_id=rule_id,
                        rule_text=text,
                        strength=float(round(alpha, 4)),
                        consequent=consequent_label,
                    )
                )
                # Mamdani min-implication: clip consequent membership function at alpha
                clipped_mf = np.fmin(alpha, mf_curve)
                # Max-aggregation across active rules
                aggregated = np.fmax(aggregated, clipped_mf)

        # Sort active rules by strength descending
        active_rules.sort(key=lambda r: r.strength, reverse=True)
        return active_rules, aggregated

    def defuzzify(self, aggregated: np.ndarray, method: str = "centroid") -> float:
        """Defuzzifies the aggregated fuzzy output using centroid method."""
        if np.sum(aggregated) == 0:
            # Default zero baseline if no rule activated
            return 0.0
        crisp = fuzz.defuzz(self.x_irr, aggregated, method)
        return float(np.round(crisp, 2))

    def evaluate(
        self,
        soil_moisture: float,
        temperature: float,
        relative_humidity: float,
        recent_rainfall: float,
        crop_stage: str = "Flowering",
    ) -> RecommendationResult:
        """End-to-end evaluation with transparent inspection of activations and curves."""
        warnings = []
        # Input validation & warning detection
        if not (0 <= soil_moisture <= 100):
            warnings.append(f"Soil moisture {soil_moisture}% is outside normal [0, 100]% bounds; clamped.")
        if not (15 <= temperature <= 48):
            warnings.append(f"Temperature {temperature}°C is outside [15, 48]°C calibrated range; clamped.")
        if not (10 <= relative_humidity <= 100):
            warnings.append(f"Relative humidity {relative_humidity}% is outside [10, 100]% bounds; clamped.")
        if recent_rainfall < 0:
            warnings.append(f"Recent rainfall {recent_rainfall}mm cannot be negative; set to 0.")

        # Compute membership activations
        memberships = self.calculate_membership_degrees(
            soil_moisture, temperature, relative_humidity, recent_rainfall
        )

        # Evaluate rules and aggregate
        active_rules, aggregated = self.evaluate_rules_and_aggregate(memberships)

        # Defuzzify via centroid
        crisp_irrigation = self.defuzzify(aggregated, method="centroid")

        # Agronomic Pearl Millet Growth Stage Adjustment:
        # Flowering / Grain Filling is the most water-critical stage (Kc ~ 1.15).
        # Vegetative (Kc ~ 0.4-0.7) and Ripening (Kc ~ 0.4-0.5) have lower sensitivity.
        stage_multipliers = {
            "Vegetative": 0.85,
            "Flowering": 1.0,     # Standard reference calibration
            "Maturity": 0.65,
            "Off-Season": 0.0,
        }
        multiplier = stage_multipliers.get(crop_stage, 1.0)
        final_depth = round(crisp_irrigation * multiplier, 2)

        # Determine qualitative category
        if final_depth < 1.0:
            category = "None"
            summary = "Adequate soil water or recent rainfall. No irrigation recommended."
        elif final_depth < 4.0:
            category = "Low"
            summary = "Mild water depletion. Light supplemental irrigation recommended (~2-3 mm)."
        elif final_depth < 7.5:
            category = "Medium"
            summary = "Moderate soil moisture deficit. Standard irrigation recommended (~5-7 mm)."
        else:
            category = "High"
            summary = "Critical dry soil and high atmospheric demand. Immediate irrigation recommended (~8-11 mm)."

        return RecommendationResult(
            irrigation_depth_mm=final_depth,
            category=category,
            linguistic_summary=summary,
            soil_moisture_pct=float(soil_moisture),
            temperature_c=float(temperature),
            relative_humidity_pct=float(relative_humidity),
            recent_rainfall_mm=float(recent_rainfall),
            crop_stage=crop_stage,
            membership_degrees=memberships,
            active_rules=active_rules,
            aggregated_fuzzy_curve=(self.x_irr, aggregated),
            warnings=warnings,
        )


if __name__ == "__main__":
    fis = PearlMilletIrrigationFIS()
    # Scenario 1: Extreme drought (dry soil, hot temp, zero rain)
    res_dry = fis.evaluate(soil_moisture=15, temperature=38, relative_humidity=25, recent_rainfall=0)
    print("Scenario Dry:", res_dry.category, f"{res_dry.irrigation_depth_mm} mm/day")
    print("Fired rules:", len(res_dry.active_rules))

    # Scenario 2: Active monsoon (wet soil, cool temp, 40mm rain)
    res_wet = fis.evaluate(soil_moisture=85, temperature=26, relative_humidity=85, recent_rainfall=40)
    print("Scenario Wet:", res_wet.category, f"{res_wet.irrigation_depth_mm} mm/day")
    print("Fired rules:", len(res_wet.active_rules))
