from functools import lru_cache

from core.config import settings
from models.schemas import (
    CascadingDisaster,
    DisasterHistory,
    FeatureCollection,
    GeoJsonGeometry,
    MapFeature,
    MapFeatureProperties,
    MapResponse,
    RegionDetailResponse,
    RegionInfo,
    RiskLegendItem,
    Shelter,
    WeatherRiskPoint,
    WeatherSummary,
)
from repositories.disaster_history_repository import (
    DisasterHistoryRepository,
    get_disaster_history_repository,
)
from repositories.region_repository import RegionRepository, get_region_repository
from repositories.shelter_repository import ShelterRepository, get_shelter_repository
from services.advisory import (
    AdviceGenerator,
    FallbackAdviceGenerator,
    GeminiAdviceGenerator,
    ResilientAdviceGenerator,
)
from services.risk_model import RISK_STYLE, RandomForestRiskPredictor, RiskPredictor
from services.terrain_service import TerrainDataError, TerrainService
from services.weather_client import (
    KmaWeatherClient,
    SampleWeatherClient,
    WeatherClient,
    WeatherClientError,
)

SHELTER_RAG_MIN_OVERALL_SCORE = 60.0
DEMO_RISK_OVERRIDES = {
    # MVP 시연용: 다른 구의 실제 분석 결과에는 영향을 주지 않습니다.
    "11320": {
        "landslide_risk_score": 70.0,
        "heavy_rain_risk_score": 70.0,
    }
}


class MapService:
    def __init__(
        self,
        repository: RegionRepository,
        risk_predictor: RiskPredictor,
        advice_generator: AdviceGenerator,
        weather_client: WeatherClient,
        fallback_weather_client: WeatherClient | None = None,
        terrain_service: TerrainService | None = None,
        history_repository: DisasterHistoryRepository | None = None,
        shelter_repository: ShelterRepository | None = None,
    ) -> None:
        self.repository = repository
        self.risk_predictor = risk_predictor
        self.advice_generator = advice_generator
        self.weather_client = weather_client
        self.fallback_weather_client = fallback_weather_client or SampleWeatherClient()
        self.terrain_service = terrain_service
        self.history_repository = history_repository
        self.shelter_repository = shelter_repository

    @staticmethod
    def _apply_demo_risk_override(region: dict, risk):
        override = DEMO_RISK_OVERRIDES.get(region["id"])
        if override is None:
            return risk

        landslide_score = override["landslide_risk_score"]
        heavy_rain_score = override["heavy_rain_risk_score"]
        cascade_score = round((landslide_score * heavy_rain_score) ** 0.5, 2)
        overall_score = round(
            landslide_score * 0.5
            + heavy_rain_score * 0.3
            + cascade_score * 0.2,
            2,
        )
        label, color = RISK_STYLE["danger"]
        return risk.model_copy(
            update={
                "overall_score": overall_score,
                "landslide_probability": landslide_score / 100.0,
                "landslide_predicted": True,
                "landslide_risk_score": landslide_score,
                "heavy_rain_risk_score": heavy_rain_score,
                "cascade_risk_score": cascade_score,
                "level": "danger",
                "label": label,
                "color": color,
                "source": "demo_override",
            }
        )

    def _enrich_region(self, source: dict) -> tuple[dict, dict]:
        region = dict(source)
        metadata: dict = {}
        if self.terrain_service is not None:
            try:
                terrain = self.terrain_service.get_features(region["center"])
                region.update(
                    landslide_map_value=terrain.landslide_map_value,
                    slope=terrain.slope,
                    elevation=terrain.elevation,
                    terrain_source=terrain.source,
                )
                metadata["landslide_sample_distance_m"] = (
                    terrain.landslide_sample_distance_m
                )
            except TerrainDataError as error:
                metadata["terrain_error"] = str(error)

        if self.history_repository is not None:
            region["disaster_history"] = self.history_repository.find_by_region(
                region["name"]
            )
        return region, metadata

    def _analyze_region(
        self, region: dict
    ) -> tuple[list[WeatherRiskPoint], str, str | None]:
        weather_error: str | None = None
        try:
            observations = self.weather_client.get_recent_observations(region)
        except WeatherClientError as error:
            weather_error = str(error)
            observations = self.fallback_weather_client.get_recent_observations(region)

        points = [
            WeatherRiskPoint(
                observed_at=observation.observed_at.isoformat(),
                temperature_c=observation.temperature_c,
                dew_point_c=observation.dew_point_c,
                humidity_percent=observation.humidity_percent,
                risk=self._apply_demo_risk_override(
                    region, self.risk_predictor.predict(region, observation)
                ),
            )
            for observation in observations
        ]
        if not points:
            raise WeatherClientError("기상 관측 데이터가 없습니다.")
        return points, observations[0].source, weather_error

    def get_map(self) -> MapResponse:
        features: list[MapFeature] = []
        weather_sources: set[str] = set()
        for source_region in self.repository.list_all():
            region, _ = self._enrich_region(source_region)
            observations, weather_source, _ = self._analyze_region(region)
            weather_sources.add(weather_source)
            risk = max(observations, key=lambda item: item.risk.overall_score).risk
            features.append(
                MapFeature(
                    id=region["id"],
                    geometry=GeoJsonGeometry(**region["geometry"]),
                    properties=MapFeatureProperties(
                        name=region["name"],
                        risk_score=risk.overall_score,
                        risk_level=risk.level,
                        risk_label=risk.label,
                        risk_color=risk.color,
                        observations=observations,
                    ),
                )
            )

        ranges = {
            "safe": (0, 19.99),
            "interest": (20, 39.99),
            "caution": (40, 59.99),
            "danger": (60, 79.99),
            "very_danger": (80, 100),
        }
        legend = [
            RiskLegendItem(
                level=level,
                label=label,
                min_score=ranges[level][0],
                max_score=ranges[level][1],
                color=color,
            )
            for level, (label, color) in RISK_STYLE.items()
        ]
        observation_date = features[0].properties.observations[0].observed_at[:10]
        return MapResponse(
            geojson=FeatureCollection(features=features),
            legend=legend,
            observation_date=observation_date,
            weather_source=(
                next(iter(weather_sources)) if len(weather_sources) == 1 else "mixed"
            ),
        )

    def get_region_detail(self, identifier_or_query: str) -> RegionDetailResponse | None:
        source_region = self.repository.find(identifier_or_query)
        if source_region is None:
            return None

        region, terrain_metadata = self._enrich_region(source_region)
        observations, weather_source, weather_error = self._analyze_region(region)
        risk = max(observations, key=lambda item: item.risk.overall_score).risk
        shelter_rag_eligible = risk.overall_score >= SHELTER_RAG_MIN_OVERALL_SCORE
        region["shelters"] = (
            self.shelter_repository.retrieve(region["name"])
            if shelter_rag_eligible and self.shelter_repository is not None
            else []
        )
        latest = observations[-1].risk.model_input
        weather_summary = WeatherSummary(
            observed_from=observations[0].observed_at,
            observed_to=observations[-1].observed_at,
            observation_count=len(observations),
            latest_rain_1h_mm=latest.rain_1h,
            latest_rain_3h_mm=latest.rain_3h,
            latest_rain_6h_mm=latest.rain_6h,
            latest_rain_12h_mm=latest.rain_12h,
            latest_rain_24h_mm=latest.rain_24h,
            latest_rain_48h_mm=latest.rain_48h,
            latest_rain_72h_mm=latest.rain_72h,
            source=weather_source,
        )
        advice = self.advice_generator.generate(region, risk, observations)
        return RegionDetailResponse(
            region=RegionInfo(id=region["id"], name=region["name"], center=region["center"]),
            risk=risk,
            weather_summary=weather_summary,
            observations=observations,
            disaster_history=[
                DisasterHistory(**item) for item in region.get("disaster_history", [])
            ],
            cascading_disasters=[
                CascadingDisaster(**item) for item in advice.cascading_disasters
            ],
            action_recommendation=advice.action_recommendation,
            shelters=[Shelter(**item) for item in region.get("shelters", [])],
            generated_by=advice.generated_by,
            metadata={
                "weather_source": weather_source,
                "weather_error": weather_error,
                "llm_error": advice.error,
                "risk_source": risk.source,
                "demo_override_applied": risk.source == "demo_override",
                "demo_override_reason": (
                    "MVP 시연을 위해 도봉구의 산사태·호우·연쇄재난 위험도를 "
                    "각 70점으로 고정"
                    if risk.source == "demo_override"
                    else None
                ),
                "risk_formula": (
                    "overall = landslide_risk*0.5 + heavy_rain_risk*0.3 "
                    "+ cascade_risk*0.2"
                ),
                "cascade_formula": "sqrt(landslide_risk * heavy_rain_risk)",
                "landslide_classification_threshold": 0.5,
                "heavy_rain_reference_mm": {
                    "rain_1h": 50,
                    "rain_3h": 90,
                    "rain_6h": 150,
                    "rain_12h": 200,
                    "rain_24h": 300,
                    "rain_48h": 400,
                    "rain_72h": 500,
                },
                "rain_accumulation_fields": {
                    "rain_1h": "rn_60m",
                    "rain_3h": "rn_03h",
                    "rain_6h": "rn_06h",
                    "rain_12h": "rn_12h",
                    "rain_24h": "rn_day (KST 자정 이후 누적)",
                    "rain_48h": "rn_02D",
                    "rain_72h": "rn_03D",
                },
                "terrain_source": region.get("terrain_source", "fallback"),
                "history_source": (
                    str(self.history_repository.csv_path)
                    if self.history_repository and self.history_repository.csv_path
                    else "fallback"
                ),
                "history_scope": "상세주소_시도 + 상세주소_시군구 정확히 일치",
                "history_record_count": len(region.get("disaster_history", [])),
                "shelter_rag_source": (
                    str(self.shelter_repository.csv_path)
                    if self.shelter_repository and self.shelter_repository.csv_path
                    else "fallback"
                ),
                "shelter_rag_retrieval": (
                    "종합 위험도 60점 이상(위험 단계)일 때 선택한 자치구와 "
                    "주소가 일치하는 "
                    "대피소를 검색하며, "
                    "CSV에 좌표가 없어 거리순이라고 단정하지 않음"
                ),
                "shelter_rag_min_overall_score": SHELTER_RAG_MIN_OVERALL_SCORE,
                "shelter_rag_eligible": shelter_rag_eligible,
                "shelter_result_count": len(region.get("shelters", [])),
                "mvp_scope": settings.mvp_region_prefix,
                "notice": (
                    "MVP 시연용으로 도봉구 위험도를 고정한 결과입니다."
                    if risk.source == "demo_override"
                    else (
                        "제공된 Random Forest Pipeline의 발생 클래스 확률을 사용했습니다."
                        if risk.source == "random_forest"
                        else "Random Forest를 사용할 수 없어 규칙 기반 결과를 사용했습니다."
                    )
                ),
                **terrain_metadata,
            },
        )


@lru_cache
def get_map_service() -> MapService:
    weather_client: WeatherClient
    if settings.kma_auth_key:
        weather_client = KmaWeatherClient(
            auth_key=settings.kma_auth_key,
            api_url=settings.kma_api_url,
            timeout_seconds=settings.kma_api_timeout_seconds,
        )
    else:
        weather_client = SampleWeatherClient()

    fallback_advice = FallbackAdviceGenerator()
    advice_generator: AdviceGenerator = fallback_advice
    if settings.gemini_api_key:
        try:
            advice_generator = ResilientAdviceGenerator(
                primary=GeminiAdviceGenerator(
                    api_key=settings.gemini_api_key,
                    model_name=settings.gemini_model,
                    timeout_seconds=settings.gemini_timeout_seconds,
                ),
                fallback=fallback_advice,
            )
        except Exception as error:
            # 선택 기능인 Gemini 초기화 실패가 지도 API 전체의 500 오류가 되지 않게 합니다.
            advice_generator = FallbackAdviceGenerator(
                initial_error=f"Gemini 초기화 실패: {error}"
            )

    return MapService(
        repository=get_region_repository(),
        risk_predictor=RandomForestRiskPredictor(settings.random_forest_model_path),
        advice_generator=advice_generator,
        weather_client=weather_client,
        fallback_weather_client=SampleWeatherClient(),
        terrain_service=TerrainService(
            settings.landslide_risk_raster_path, settings.dem_raster_path
        ),
        history_repository=get_disaster_history_repository(),
        shelter_repository=get_shelter_repository(),
    )
