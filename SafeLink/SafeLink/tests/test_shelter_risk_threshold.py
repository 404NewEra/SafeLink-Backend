from datetime import datetime

from models.schemas import RandomForestInput, RiskAnalysis, RiskWeights
from repositories.region_repository import get_region_repository
from services.advisory import FallbackAdviceGenerator
from services.map_service import MapService, SHELTER_RAG_MIN_OVERALL_SCORE
from services.weather_client import KST, WeatherObservation


class FixedWeatherClient:
    def get_recent_observations(self, region, now=None):
        return [
            WeatherObservation(
                observed_at=datetime(2026, 9, 17, 14, 0, tzinfo=KST),
                temperature_c=23,
                dew_point_c=20,
                humidity_percent=90,
                rain_1h=0,
                rain_3h=0,
                rain_6h=0,
                rain_12h=0,
                rain_24h=0,
                rain_48h=0,
                rain_72h=0,
                source="test",
            )
        ]


class FixedRiskPredictor:
    def __init__(self, overall_score: float) -> None:
        self.overall_score = overall_score

    def predict(self, region, weather):
        level = "danger" if self.overall_score >= 60 else "caution"
        return RiskAnalysis(
            overall_score=self.overall_score,
            landslide_probability=0.5,
            landslide_predicted=True,
            landslide_risk_score=50,
            heavy_rain_risk_score=50,
            cascade_risk_score=50,
            weights=RiskWeights(landslide=0.5, heavy_rain=0.3, cascade=0.2),
            level=level,
            label="위험" if level == "danger" else "주의",
            color="#E67E22" if level == "danger" else "#F1C40F",
            model_input=RandomForestInput(
                rain_1h=0,
                rain_3h=0,
                rain_6h=0,
                rain_12h=0,
                rain_24h=0,
                rain_48h=0,
                rain_72h=0,
                slope=0,
                elevation=0,
                landslide_map_value=5,
            ),
            source="fallback",
        )


class RecordingShelterRepository:
    csv_path = None

    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, region_name):
        self.calls += 1
        return [
            {
                "id": "s1",
                "name": "테스트 대피소",
                "address": "서울특별시 관악구 테스트로 1",
            }
        ]


def _service(score: float, shelter_repository: RecordingShelterRepository) -> MapService:
    return MapService(
        repository=get_region_repository(),
        risk_predictor=FixedRiskPredictor(score),
        advice_generator=FallbackAdviceGenerator(),
        weather_client=FixedWeatherClient(),
        shelter_repository=shelter_repository,
    )


def test_shelter_rag_is_skipped_below_danger_level() -> None:
    shelters = RecordingShelterRepository()

    detail = _service(SHELTER_RAG_MIN_OVERALL_SCORE - 0.01, shelters).get_region_detail(
        "관악구"
    )

    assert detail is not None
    assert shelters.calls == 0
    assert detail.shelters == []
    assert detail.metadata["shelter_rag_eligible"] is False


def test_shelter_rag_starts_at_danger_level() -> None:
    shelters = RecordingShelterRepository()

    detail = _service(SHELTER_RAG_MIN_OVERALL_SCORE, shelters).get_region_detail(
        "관악구"
    )

    assert detail is not None
    assert shelters.calls == 1
    assert detail.shelters[0].name == "테스트 대피소"
    assert detail.metadata["shelter_rag_eligible"] is True
    assert "테스트 대피소" in detail.action_recommendation
