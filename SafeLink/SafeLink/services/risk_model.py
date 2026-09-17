from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from models.schemas import RandomForestInput, RiskAnalysis, RiskWeights
from services.weather_client import WeatherObservation

RISK_STYLE = {
    "safe": ("안전", "#2ECC71"),
    "interest": ("관심", "#3498DB"),
    "caution": ("주의", "#F1C40F"),
    "danger": ("위험", "#E67E22"),
    "very_danger": ("매우 위험", "#E74C3C"),
}

LANDSLIDE_WEIGHT = 0.5
HEAVY_RAIN_WEIGHT = 0.3
CASCADE_WEIGHT = 0.2
CLASSIFICATION_THRESHOLD = 0.5
MODEL_FEATURE_NAMES = [
    "rain_1h",
    "rain_3h",
    "rain_6h",
    "rain_12h",
    "rain_24h",
    "rain_48h",
    "rain_72h",
    "slope",
    "elevation",
    "landslide_map_value",
]


class RiskPredictor(Protocol):
    def predict(
        self,
        region: dict[str, Any],
        weather: WeatherObservation,
    ) -> RiskAnalysis:
        """지역·강수 특성으로 위험도를 예측합니다."""


@dataclass(frozen=True)
class ModelFeatures:
    rain_1h: float
    rain_3h: float
    rain_6h: float
    rain_12h: float
    rain_24h: float
    rain_48h: float
    rain_72h: float
    slope: float
    elevation: float
    landslide_map_value: int

    def as_vector(self) -> list[float]:
        """Random Forest 학습 당시와 동일한 순서의 입력 벡터입니다."""
        return [
            self.rain_1h,
            self.rain_3h,
            self.rain_6h,
            self.rain_12h,
            self.rain_24h,
            self.rain_48h,
            self.rain_72h,
            self.slope,
            self.elevation,
            float(self.landslide_map_value),
        ]

    def as_record(self) -> dict[str, float]:
        return dict(zip(MODEL_FEATURE_NAMES, self.as_vector()))

    def to_schema(self) -> RandomForestInput:
        return RandomForestInput(**self.__dict__)


class WeatherPreprocessor:
    def transform(
        self,
        region: dict[str, Any],
        weather: WeatherObservation,
    ) -> ModelFeatures:
        return ModelFeatures(
            rain_1h=round(max(0.0, weather.rain_1h), 3),
            rain_3h=round(max(0.0, weather.rain_3h), 3),
            rain_6h=round(max(0.0, weather.rain_6h), 3),
            rain_12h=round(max(0.0, weather.rain_12h), 3),
            rain_24h=round(max(0.0, weather.rain_24h), 3),
            rain_48h=round(max(0.0, weather.rain_48h), 3),
            rain_72h=round(max(0.0, weather.rain_72h), 3),
            slope=max(0.0, min(90.0, float(region["slope"]))),
            elevation=float(region["elevation"]),
            landslide_map_value=max(
                1, min(5, int(region["landslide_map_value"]))
            ),
        )


def _level_for_score(score: float) -> str:
    if score < 20:
        return "safe"
    if score < 40:
        return "interest"
    if score < 60:
        return "caution"
    if score < 80:
        return "danger"
    return "very_danger"


def _heavy_rain_score(features: ModelFeatures) -> float:
    """각 누적 시간대의 기준량 도달 비율로 호우 위험도를 0~100으로 환산합니다."""
    normalized = (
        min(features.rain_1h / 50.0, 1.0) * 0.20
        + min(features.rain_3h / 90.0, 1.0) * 0.15
        + min(features.rain_6h / 150.0, 1.0) * 0.15
        + min(features.rain_12h / 200.0, 1.0) * 0.15
        + min(features.rain_24h / 300.0, 1.0) * 0.15
        + min(features.rain_48h / 400.0, 1.0) * 0.10
        + min(features.rain_72h / 500.0, 1.0) * 0.10
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
    """저장된 분류 Pipeline의 발생 클래스 확률을 산사태 위험도로 사용합니다."""

    def __init__(
        self,
        model_path: str,
        preprocessor: WeatherPreprocessor | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.preprocessor = preprocessor or WeatherPreprocessor()
        if model is None:
            path = Path(model_path)
            if not path.is_file():
                raise FileNotFoundError(f"Random Forest 모델 파일이 없습니다: {path}")
            import joblib

            model = joblib.load(path)
        self.model = model
        actual_names = [str(value) for value in getattr(self.model, "feature_names_in_", [])]
        if actual_names != MODEL_FEATURE_NAMES:
            raise ValueError(
                "Random Forest 입력 스키마 불일치: "
                f"expected={MODEL_FEATURE_NAMES}, actual={actual_names}"
            )
        if list(getattr(self.model, "classes_", [])) != [0, 1]:
            raise ValueError(
                f"Random Forest 클래스가 [0, 1]이 아닙니다: {self.model.classes_}"
            )

    def predict(
        self,
        region: dict[str, Any],
        weather: WeatherObservation,
    ) -> RiskAnalysis:
        features = self.preprocessor.transform(region, weather)
        # 학습 Pipeline의 feature_names_in_과 동일한 10개 피처 순서를 사용합니다.
        import pandas as pd

        model_input = pd.DataFrame([features.as_record()], columns=MODEL_FEATURE_NAMES)
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
        probability = float(self.model.predict_proba(model_input)[0][positive_index])
        return _build_analysis(features, probability, "random_forest")
