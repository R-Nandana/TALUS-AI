"""
TALUS Physics Engine — Slope Stability Calculation Module
==========================================================
Provides simplified Limit Equilibrium / Infinite Slope stability analysis for open-pit mine slopes.

Formulation & Methodology:
--------------------------
1. Extended Infinite Slope Model with Pore Pressure & Dynamic Loading:
   - Mohr-Coulomb failure criterion: tau_f = c' + sigma_n' * tan(phi')
   - Total Normal Stress: sigma_n = (gamma * H + q) * cos^2(beta)
   - Effective Normal Stress: sigma_n' = max(0.1, sigma_n - u)
   - Shear Mobilized (Driving): tau_m = (gamma * H + q) * (sin(beta) * cos(beta) + k_c)
   - Factor of Safety (FoS): FoS = tau_f / max(0.01, tau_m)

Where:
  H = Slope height (m)
  beta = Slope angle (radians)
  c' = Effective cohesion (kPa)
  phi' = Effective internal friction angle (radians)
  gamma = Rock mass unit weight (kN/m^3)
  u = Pore water pressure (kPa)
  q = Equipment surcharge load (kPa)
  k_c = Blast vibration seismic pseudo-static coefficient (dimensionless, derived from blast intensity)

Assumptions & Limitations:
---------------------------
- Planar translational slip surface assumed parallel to slope surface at depth H.
- Uniform isotropic slope geometry within sector observation depth.
- Simplified pseudo-static approximation for blasting vibration effects.
- Prototype indicator intended for preliminary safety assessment, not replacing 3D numerical modelling or certified geotechnical engineering audits.
"""

import math
from typing import Dict, Any, Tuple


def calculate_factor_of_safety(
    slope_height_m: float,
    slope_angle_deg: float,
    cohesion_kpa: float,
    friction_angle_deg: float,
    unit_weight_kn_m3: float,
    pore_pressure_kpa: float,
    recent_blasting: int = 0,
    blast_intensity: float = 0.0,
    equipment_load: float = 0.0,
) -> float:
    """
    Calculates the physics-derived Factor of Safety (FoS) for a slope sector.
    Returns float rounded to 3 decimal places.
    """
    # Guard against invalid negative values
    H = max(1.0, float(slope_height_m))
    beta = math.radians(max(5.0, min(85.0, float(slope_angle_deg))))
    c_prime = max(0.0, float(cohesion_kpa))
    phi_prime = math.radians(max(1.0, min(75.0, float(friction_angle_deg))))
    gamma = max(10.0, float(unit_weight_kn_m3))
    u = max(0.0, float(pore_pressure_kpa))
    q = max(0.0, float(equipment_load))
    
    # Pseudo-static blast vibration coefficient (k_c)
    k_c = 0.015 * float(blast_intensity) if recent_blasting == 1 else 0.0
    
    # Weight per unit surface area of slope slice
    weight_term = (gamma * H) + q
    
    # Normal stress on plane
    sigma_n = weight_term * (math.cos(beta) ** 2)
    
    # Effective normal stress (accounting for pore water pressure u)
    sigma_n_effective = max(0.1, sigma_n - u)
    
    # Mohr-Coulomb shear strength available
    tau_strength = c_prime + (sigma_n_effective * math.tan(phi_prime))
    
    # Driving shear stress mobilized
    tau_driving = weight_term * ((math.sin(beta) * math.cos(beta)) + k_c)
    
    if tau_driving <= 0.001:
        fos = 5.0
    else:
        fos = tau_strength / tau_driving
        
    return round(float(fos), 3)


def categorize_physics_stability(fos: float) -> Tuple[str, str]:
    """
    Maps Factor of Safety (FoS) to geotechnical stability category & risk state.
    Returns (Category, Description)
    """
    if fos < 1.0:
        return "Critical", "Statistically Unstable — Driving stress exceeds available shear strength (FoS < 1.0)"
    elif fos < 1.3:
        return "High Risk", "Marginally Stable — Inadequate safety factor for mining operations (1.0 <= FoS < 1.3)"
    elif fos < 1.5:
        return "Moderate Risk", "Conditionally Acceptable — Moderate stability margin, observe under heavy rainfall (1.3 <= FoS < 1.5)"
    else:
        return "Low Risk", "Stable Slope Sector — Adequate safety factor exceeding standard threshold (FoS >= 1.5)"


def run_physics_assessment(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes complete physics assessment on raw sector dictionary.
    """
    fos = calculate_factor_of_safety(
        slope_height_m=input_data.get("slope_height_m", 50.0),
        slope_angle_deg=input_data.get("slope_angle_deg", 45.0),
        cohesion_kpa=input_data.get("cohesion_kpa", 35.0),
        friction_angle_deg=input_data.get("friction_angle_deg", 30.0),
        unit_weight_kn_m3=input_data.get("unit_weight_kn_m3", 24.0),
        pore_pressure_kpa=input_data.get("pore_pressure_kpa", 40.0),
        recent_blasting=int(input_data.get("recent_blasting", 0)),
        blast_intensity=float(input_data.get("blast_intensity", 0.0)),
        equipment_load=float(input_data.get("equipment_load", 0.0)),
    )
    
    category, description = categorize_physics_stability(fos)
    
    return {
        "factor_of_safety": fos,
        "physics_category": category,
        "physics_description": description,
        "formulation": "Limit Equilibrium Infinite Slope Model with Pore Pressure & Seismic Surcharge"
    }


if __name__ == "__main__":
    test_input = {
        "slope_height_m": 60,
        "slope_angle_deg": 45,
        "cohesion_kpa": 30,
        "friction_angle_deg": 28,
        "unit_weight_kn_m3": 25,
        "pore_pressure_kpa": 120,
        "recent_blasting": 1,
        "blast_intensity": 6.5,
        "equipment_load": 40.0,
    }
    res = run_physics_assessment(test_input)
    print("Physics Assessment Test Result:")
    print(res)
