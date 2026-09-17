from typing import Any, Literal

from pydantic import BaseModel, Field


class RiskAnalysis(BaseModel):
    overall_score: float = Field(ge=0, le=100, description="종합 위험도 점수")
    level: Literal["low", "moderate", "high", "critical"]
    label: str
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    rainfall_mm: float = Field(ge=0)
    slope_degree: float = Field(ge=0, le=90)
    landslide_grade: int = Field(ge=1, le=5)
    source: Literal["fallback", "random_forest"]


class GeoJsonGeometry(BaseModel):
    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]


class MapFeatureProperties(BaseModel):
    name: str
    risk_score: float = Field(ge=0, le=100)
    risk_level: str
    risk_label: str
    risk_color: str


class MapFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    id: str
    geometry: GeoJsonGeometry
    properties: MapFeatureProperties


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[MapFeature]


class RiskLegendItem(BaseModel):
    level: str
    label: str
    min_score: float
    max_score: float
    color: str


class MapResponse(BaseModel):
    geojson: FeatureCollection
    legend: list[RiskLegendItem]


class DisasterHistory(BaseModel):
    date: str
    type: str
    description: str


class CascadingDisaster(BaseModel):
    type: str
    possibility: str
    description: str


class Shelter(BaseModel):
    id: str
    name: str
    address: str
    latitude: float
    longitude: float
    capacity: int | None = Field(default=None, ge=0)
    phone: str | None = None


class RegionInfo(BaseModel):
    id: str
    name: str
    center: list[float] = Field(description="[경도, 위도]")


class RegionDetailResponse(BaseModel):
    region: RegionInfo
    risk: RiskAnalysis
    disaster_history: list[DisasterHistory]
    cascading_disasters: list[CascadingDisaster]
    action_recommendation: str
    shelters: list[Shelter]
    generated_by: Literal["fallback", "llm"]
    metadata: dict[str, Any] = Field(default_factory=dict)
