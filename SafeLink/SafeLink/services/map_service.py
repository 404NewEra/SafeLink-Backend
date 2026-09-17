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
        if self.shelter_repository is not None:
            region["shelters"] = self.shelter_repository.retrieve(region["name"])
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
                risk=self.risk_predictor.predict(region, observation),
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
            "low": (0, 24.99),
            "moderate": (25, 49.99),
            "high": (50, 74.99),
            "critical": (75, 100),
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
        latest = observations[-1].risk.model_input
        weather_summary = WeatherSummary(
            observed_from=observations[0].observed_at,
            observed_to=observations[-1].observed_at,
            observation_count=len(observations),
            latest_rain_15m_mm=latest.rain_15m,
            latest_rain_60m_mm=latest.rain_60m,
            latest_rain_3h_mm=latest.rain_3h,
            latest_rain_6h_mm=latest.rain_6h,
            latest_rain_24h_mm=latest.rain_24h,
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
                "risk_formula": (
                    "overall = landslide_risk*0.5 + heavy_rain_risk*0.3 "
                    "+ cascade_risk*0.2"
                ),
                "cascade_formula": "sqrt(landslide_risk * heavy_rain_risk)",
                "landslide_classification_threshold": 0.5,
                "heavy_rain_reference_mm": {
                    "rain_15m": 20,
                    "rain_60m": 50,
                    "rain_3h": 90,
                    "rain_6h": 150,
                    "rain_24h": 300,
                },
                "rain_24h_method": "기상청 rn_day(KST 자정 이후 누적)를 MVP의 rain_24h로 사용",
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
                "shelter_rag_retrieval": "지역 구 이름이 대피소 주소에 일치하는 상위 CSV 행",
                "shelter_result_count": len(region.get("shelters", [])),
                "mvp_scope": settings.mvp_region_prefix,
                "notice": (
                    "제공된 Random Forest Pipeline의 발생 클래스 확률을 사용했습니다."
                    if risk.source == "random_forest"
                    else "Random Forest를 사용할 수 없어 규칙 기반 결과를 사용했습니다."
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
