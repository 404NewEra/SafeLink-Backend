from typing import Any, Protocol

from models.schemas import RiskAnalysis

RISK_STYLE = {
    "low": ("낮음", "#2ECC71"),
    "moderate": ("보통", "#F1C40F"),
    "high": ("높음", "#E67E22"),
    "critical": ("매우 높음", "#E74C3C"),
}


class RiskPredictor(Protocol):
    def predict(self, region: dict[str, Any]) -> RiskAnalysis:
        """지역 특성으로 위험도를 예측합니다."""


def _level_for_score(score: float) -> str:
    if score < 25:
        return "low"
    if score < 50:
        return "moderate"
    if score < 75:
        return "high"
    return "critical"


class FallbackRiskPredictor:
    """실제 모델 연결 전 개발용 예측기입니다.

    regions.json의 sample_risk_score만 사용하므로 운영 위험 판단에 사용하면 안 됩니다.
    """

    def predict(self, region: dict[str, Any]) -> RiskAnalysis:
        score = float(region["sample_risk_score"])
        level = _level_for_score(score)
        label, color = RISK_STYLE[level]
        return RiskAnalysis(
            overall_score=score,
            level=level,
            label=label,
            color=color,
            rainfall_mm=region["rainfall_mm"],
            slope_degree=region["slope_degree"],
            landslide_grade=region["landslide_grade"],
            source="fallback",
        )


class RandomForestRiskPredictor:
    """학습된 Random Forest를 연결할 자리입니다."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        # TODO: 모델 추가 후 joblib.load(model_path)로 self.model을 초기화하세요.
        # TODO: 학습 당시 feature 순서/전처리 파이프라인도 함께 저장해 사용하세요.
        raise NotImplementedError("Random Forest 모델 연결이 아직 구현되지 않았습니다.")

    def predict(self, region: dict[str, Any]) -> RiskAnalysis:
        # TODO: rainfall_mm, slope_degree, landslide_grade 등을 feature로 구성하고
        #       predict/predict_proba 결과를 0~100 점수와 위험 단계로 변환하세요.
        raise NotImplementedError
