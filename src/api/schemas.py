from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    vehicle_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float]
    class_id: int


class VehicleCounts(BaseModel):
    total_vehicles: int
    cars: int
    buses: int
    trucks: int
    motorcycles: int
    bicycles: int


class DensityResult(BaseModel):
    vehicle_count: int
    density: str
    density_level: str
    color: str
    threshold_range: str


class ImagePredictionResponse(BaseModel):
    status: str = "success"
    source: str
    detections: List[Detection]
    counts: VehicleCounts
    density: DensityResult
    congestion_index: float
    annotated_image_base64: Optional[str] = None


class VideoPredictionResponse(BaseModel):
    status: str = "success"
    source: str
    total_frames: int
    summary: Dict
    session_analytics: Dict


class AnalyticsResponse(BaseModel):
    status: str = "success"
    metrics: Dict
    model_name: str


class DensityConfigUpdate(BaseModel):
    low_max: int = Field(default=10, ge=1)
    medium_max: int = Field(default=25, ge=2)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
