from functools import lru_cache

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
)
from repositories.region_repository import RegionRepository, get_region_repository
from services.advisory import AdviceGenerator, FallbackAdviceGenerator
from services.risk_model import FallbackRiskPredictor, RISK_STYLE, RiskPredictor


class MapService:
    def __init__(
        self,
        repository: RegionRepository,
        risk_predictor: RiskPredictor,
        advice_generator: AdviceGenerator,
    ) -> None:
        self.repository = repository
        self.risk_predictor = risk_predictor
        self.advice_generator = advice_generator

    def get_map(self) -> MapResponse:
        features: list[MapFeature] = []
        for region in self.repository.list_all():
            risk = self.risk_predictor.predict(region)
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
        return MapResponse(geojson=FeatureCollection(features=features), legend=legend)

    def get_region_detail(self, identifier_or_query: str) -> RegionDetailResponse | None:
        region = self.repository.find(identifier_or_query)
        if region is None:
            return None

        risk = self.risk_predictor.predict(region)
        recommendation, generated_by = self.advice_generator.generate(region, risk)
        return RegionDetailResponse(
            region=RegionInfo(id=region["id"], name=region["name"], center=region["center"]),
            risk=risk,
            disaster_history=[DisasterHistory(**item) for item in region["disaster_history"]],
            cascading_disasters=[
                CascadingDisaster(**item) for item in region["cascading_disasters"]
            ],
            action_recommendation=recommendation,
            shelters=[Shelter(**item) for item in region["shelters"]],
            generated_by=generated_by,
            metadata={
                "notice": "현재 위험도와 행동 추천은 개발용 대체 구현 결과입니다.",
            },
        )


@lru_cache
def get_map_service() -> MapService:
    # TODO: 모델 파일이 준비되면 FallbackRiskPredictor를
    # RandomForestRiskPredictor(settings.random_forest_model_path)로 교체하세요.
    # TODO: LLM 정보가 준비되면 FallbackAdviceGenerator를
    # LangChainAdviceGenerator(settings.llm_api_key, settings.llm_model)로 교체하세요.
    return MapService(
        repository=get_region_repository(),
        risk_predictor=FallbackRiskPredictor(),
        advice_generator=FallbackAdviceGenerator(),
    )
