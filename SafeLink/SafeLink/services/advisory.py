from typing import Any, Protocol

from models.schemas import RiskAnalysis


class AdviceGenerator(Protocol):
    def generate(self, region: dict[str, Any], risk: RiskAnalysis) -> tuple[str, str]:
        """(행동 추천 문구, 생성 방식)을 반환합니다."""


class FallbackAdviceGenerator:
    def generate(self, region: dict[str, Any], risk: RiskAnalysis) -> tuple[str, str]:
        recommendations = {
            "low": "기상 정보를 주기적으로 확인하고 가까운 대피소 위치를 미리 확인하세요.",
            "moderate": "비탈면과 하천 주변 접근을 줄이고 재난 알림을 켜 두세요.",
            "high": "외출을 자제하고 대피 준비물을 챙긴 뒤 지자체 안내에 따라 이동하세요.",
            "critical": "즉시 위험 지역을 벗어나 가장 가까운 대피소로 이동하고 공식 안내를 따르세요.",
        }
        return recommendations[risk.level], "fallback"


class LangChainAdviceGenerator:
    """LangChain과 원하는 LLM을 연결할 자리입니다."""

    def __init__(self, api_key: str, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name
        # TODO: 사용할 공급자의 ChatModel과 PromptTemplate/chain을 초기화하세요.
        # API 키는 코드에 넣지 말고 LLM_API_KEY 환경 변수로 주입하세요.
        raise NotImplementedError("LangChain/LLM 연결이 아직 구현되지 않았습니다.")

    def generate(self, region: dict[str, Any], risk: RiskAnalysis) -> tuple[str, str]:
        # TODO: 위험 분석, 과거 재난 이력, 연쇄 재난 정보를 프롬프트에 전달하세요.
        # TODO: LLM 장애 시 FallbackAdviceGenerator를 사용하도록 예외 처리를 추가하세요.
        raise NotImplementedError
