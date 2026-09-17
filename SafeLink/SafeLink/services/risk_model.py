from dataclasses import dataclass
from typing import Any, Protocol

from models.schemas import RandomForestInput, RiskAnalysis, RiskWeights
from services.weather_client import WeatherObservation

RISK_STYLE = {
    "low": ("낮음", "#2ECC71"),
    "moderate": ("보통", "#F1C40F"),
    "high": ("높음", "#E67E22"),
    "critical": ("매우 높음", "#E74C3C"),
}

LANDSLIDE_WEIGHT = 0.5
HEAVY_RAIN_WEIGHT = 0.3
CASCADE_WEIGHT = 0.2
CLASSIFICATION_THRESHOLD = 0.5


class RiskPredictor(Protocol):
    def predict(
        self,
        region: dict[str, Any],
        weather: WeatherObservation,
    ) -> RiskAnalysis:
        """지역·강수 특성으로 위험도를 예측합니다."""


@dataclass(frozen=True)
class ModelFeatures:
    landslide_map_value: int
    rain_15m: float
    rain_60m: float
    rain_3h: float
    rain_6h: float
    rain_24h: float
    slope: float
    elevation: float

    def as_vector(self) -> list[float]:
        """Random Forest 학습 당시와 동일한 순서의 입력 벡터입니다."""
        return [
            float(self.landslide_map_value),
            self.rain_15m,
            self.rain_60m,
            self.rain_3h,
            self.rain_6h,
            self.rain_24h,
            self.slope,
            self.elevation,
        ]

    def to_schema(self) -> RandomForestInput:
        return RandomForestInput(**self.__dict__)


class WeatherPreprocessor:
    def transform(
        self,
        region: dict[str, Any],
        weather: WeatherObservation,
    ) -> ModelFeatures:
        return ModelFeatures(
            landslide_map_value=max(
                1, min(5, int(region["landslide_map_value"]))
            ),
            rain_15m=round(max(0.0, weather.rain_15m), 3),
            rain_60m=round(max(0.0, weather.rain_60m), 3),
            rain_3h=round(max(0.0, weather.rain_3h), 3),
            rain_6h=round(max(0.0, weather.rain_6h), 3),
            rain_24h=round(max(0.0, weather.rain_24h), 3),
            slope=max(0.0, min(90.0, float(region["slope"]))),
            elevation=float(region["elevation"]),
        )


def _level_for_score(score: float) -> str:
    if score < 25:
        return "low"
    if score < 50:
        return "moderate"
    if score < 75:
        return "high"
    return "critical"


def _heavy_rain_score(features: ModelFeatures) -> float:
    """각 누적 시간대의 기준량 도달 비율로 호우 위험도를 0~100으로 환산합니다."""
    normalized = (
        min(features.rain_15m / 20.0, 1.0) * 0.15
        + min(features.rain_60m / 50.0, 1.0) * 0.25
        + min(features.rain_3h / 90.0, 1.0) * 0.20
        + min(features.rain_6h / 150.0, 1.0) * 0.15
        + min(features.rain_24h / 300.0, 1.0) * 0.25
    )
    return round(normalized * 100.0, 2)


def _build_analysis(
    features: ModelFeatures,
    landslide_probability: float,
    source: str,
) -> RiskAnalysis:
    probability = max(0.0, min(1.0, landslide_probability))
    landslide_score = round(probability * 100.0, 2)
    heavy_rain_score = _heavy_rain_score(features)
    # 기하평균은 두 위험도가 모두 높을 때만 크게 증가하므로 연계 위험 표현에 적합합니다.
    cascade_score = round((landslide_score * heavy_rain_score) ** 0.5, 2)
    overall_score = round(
        landslide_score * LANDSLIDE_WEIGHT
        + heavy_rain_score * HEAVY_RAIN_WEIGHT
        + cascade_score * CASCADE_WEIGHT,
        2,
    )
    level = _level_for_score(overall_score)
    label, color = RISK_STYLE[level]
    return RiskAnalysis(
        overall_score=overall_score,
        landslide_probability=round(probability, 6),
        landslide_predicted=probability >= CLASSIFICATION_THRESHOLD,
        landslide_risk_score=landslide_score,
        heavy_rain_risk_score=heavy_rain_score,
        cascade_risk_score=cascade_score,
        weights=RiskWeights(
            landslide=LANDSLIDE_WEIGHT,
            heavy_rain=HEAVY_RAIN_WEIGHT,
            cascade=CASCADE_WEIGHT,
        ),
        level=level,
        label=label,
        color=color,
        model_input=features.to_schema(),
        source=source,
    )


class FallbackRiskPredictor:
    """실제 모델 연결 전 개발용 규칙 기반 예측기입니다."""

    def __init__(self, preprocessor: WeatherPreprocessor | None = None) -> None:
        self.preprocessor = preprocessor or WeatherPreprocessor()

    def predict(
        self,
        region: dict[str, Any],
        weather: WeatherObservation,
    ) -> RiskAnalysis:
        features = self.preprocessor.transform(region, weather)
        heavy_rain_score = _heavy_rain_score(features)
        fallback_landslide_score = min(
            100.0,
            ((6 - features.landslide_map_value) / 5.0 * 35.0)
            + (features.slope / 90.0 * 25.0)
            + (min(max(features.elevation, 0.0), 1000.0) / 1000.0 * 10.0)
            + (heavy_rain_score * 0.30),
        )
        return _build_analysis(
            features=features,
            landslide_probability=fallback_landslide_score / 100.0,
            source="fallback",
        )


class RandomForestRiskPredictor:
    """학습된 Random Forest 모델을 연결할 예측기입니다."""

    def __init__(
        self, model_path: str, preprocessor: WeatherPreprocessor | None = None
    ) -> None:
        self.model_path = model_path
        self.preprocessor = preprocessor or WeatherPreprocessor()
        self.model = None
        # Random Forest 파일을 추가한 뒤 아래 주석을 해제하세요.
        # import joblib
        # self.model = joblib.load(model_path)

    def predict(
        self,
        region: dict[str, Any],
        weather: WeatherObservation,
    ) -> RiskAnalysis:
        if self.model is None:
            raise RuntimeError("Random Forest 모델 파일 연결이 아직 활성화되지 않았습니다.")

        features = self.preprocessor.transform(region, weather)
        # 입력 순서:
        # landslide_map_value, rain_15m, rain_60m, rain_3h,
        # rain_6h, rain_24h, slope, elevation
        vector = [features.as_vector()]
        classes = list(getattr(self.model, "classes_", []))
        positive_index = next(
            (
                index
                for index, class_value in enumerate(classes)
                if class_value in (1, True, "1")
            ),
            None,
        )
        if positive_index is None:
            raise RuntimeError("Random Forest classes_에서 산사태 발생 클래스 1을 찾지 못했습니다.")
        probability = float(self.model.predict_proba(vector)[0][positive_index])
        return _build_analysis(features, probability, "random_forest")
