from typing import Any, Literal

from pydantic import BaseModel, Field


class RandomForestInput(BaseModel):
    landslide_map_value: int = Field(ge=1, le=5)
    rain_15m: float = Field(ge=0)
    rain_60m: float = Field(ge=0)
    rain_3h: float = Field(ge=0)
    rain_6h: float = Field(ge=0)
    rain_24h: float = Field(ge=0)
    slope: float = Field(ge=0, le=90)
    elevation: float


class RiskAnalysis(BaseModel):
    overall_score: float = Field(ge=0, le=100, description="종합 위험도 점수")
    level: Literal["low", "moderate", "high", "critical"]
    label: str
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    model_input: RandomForestInput
    source: Literal["fallback", "random_forest"]


class WeatherRiskPoint(BaseModel):
    observed_at: str = Field(description="KST 기준 ISO 8601 관측 시각")
    temperature_c: float | None = None
    dew_point_c: float | None = None
    humidity_percent: float | None = Field(default=None, ge=0, le=100)
    risk: RiskAnalysis


class GeoJsonGeometry(BaseModel):
    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]


class MapFeatureProperties(BaseModel):
    name: str
    risk_score: float = Field(ge=0, le=100)
    risk_level: str
    risk_label: str
    risk_color: str
    observations: list[WeatherRiskPoint]


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
    observation_date: str
    weather_source: str


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


class WeatherSummary(BaseModel):
    observed_from: str
    observed_to: str
    observation_count: int = Field(ge=1)
    latest_rain_15m_mm: float = Field(ge=0)
    latest_rain_60m_mm: float = Field(ge=0)
    latest_rain_3h_mm: float = Field(ge=0)
    latest_rain_6h_mm: float = Field(ge=0)
    latest_rain_24h_mm: float = Field(ge=0)
    source: str


class RegionDetailResponse(BaseModel):
    region: RegionInfo
    risk: RiskAnalysis
    weather_summary: WeatherSummary
    observations: list[WeatherRiskPoint]
    disaster_history: list[DisasterHistory]
    cascading_disasters: list[CascadingDisaster]
    action_recommendation: str
    shelters: list[Shelter]
    generated_by: Literal["fallback", "llm"]
    metadata: dict[str, Any] = Field(default_factory=dict)
