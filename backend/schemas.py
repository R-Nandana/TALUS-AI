"""Pydantic request/response contracts for TALUS REST API."""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict

class SectorPredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "sector_id":"S1","slope_height_m":65,"slope_angle_deg":38,"cohesion_kpa":45,
        "friction_angle_deg":34,"unit_weight_kn_m3":25,"rainfall_mm":12,"groundwater_level":8,
        "pore_pressure_kpa":35,"deformation_rate_mm_day":2.1,"displacement_mm":14,
        "recent_blasting":0,"blast_intensity":0,"equipment_load":15,"scenario_name":"Assessment"
    }})
    sector_id: str = Field(..., description="Sector identifier")
    slope_height_m: float = Field(..., ge=1.0, le=300.0)
    slope_angle_deg: float = Field(..., ge=5.0, le=85.0)
    cohesion_kpa: float = Field(..., ge=0.0, le=500.0)
    friction_angle_deg: float = Field(..., ge=1.0, le=75.0)
    unit_weight_kn_m3: float = Field(..., ge=10.0, le=40.0)
    rainfall_mm: float = Field(..., ge=0.0, le=500.0)
    groundwater_level: float = Field(..., ge=0.0, le=100.0)
    pore_pressure_kpa: float = Field(..., ge=0.0, le=1000.0)
    deformation_rate_mm_day: float = Field(..., ge=0.0, le=200.0)
    displacement_mm: float = Field(..., ge=0.0, le=2000.0)
    recent_blasting: int = Field(0, ge=0, le=1)
    blast_intensity: float = Field(0.0, ge=0.0, le=10.0)
    equipment_load: float = Field(0.0, ge=0.0, le=300.0)
    scenario_name: Optional[str] = Field("User Prediction")

class FeatureContributor(BaseModel):
    feature: str
    value: float
    impact_score: float
    direction: str

class SectorPredictResponse(BaseModel):
    sector_id: str
    timestamp: str
    factor_of_safety: float
    physics_category: str
    physics_description: str
    ml_risk_label: str
    ml_risk_probability: float
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    decision_recommendation: str
    top_contributors: List[FeatureContributor]
    saved_db_id: int
    formulation: str
    disclaimer: str
