import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from models.schemas import RiskAnalysis, WeatherRiskPoint


class AdviceGenerationError(RuntimeError):
    """LangChain 또는 LLM의 상세 분석 생성이 실패했습니다."""


@dataclass(frozen=True)
class AdviceResult:
    cascading_disasters: list[dict[str, str]]
    action_recommendation: str
    generated_by: Literal["fallback", "llm"]
    error: str | None = None


class LlmCascadingDisaster(BaseModel):
    type: str = Field(description="예상되는 연쇄 재난 유형")
    possibility: str = Field(description="낮음, 보통, 높음, 매우 높음 중 하나")
    description: str = Field(description="입력 데이터에 근거한 짧은 설명")


class LlmAdviceOutput(BaseModel):
    cascading_disasters: list[LlmCascadingDisaster] = Field(
        description="입력 데이터로 예상되는 연쇄 재난. 근거가 없으면 빈 배열"
    )
    action_recommendation: str = Field(
        description="사용자가 바로 수행할 수 있는 한국어 행동 추천"
    )


class AdviceGenerator(Protocol):
    def generate(
        self,
        region: dict[str, Any],
        risk: RiskAnalysis,
        observations: list[WeatherRiskPoint],
    ) -> AdviceResult:
        """연쇄 재난 예측과 행동 추천을 반환합니다."""


class FallbackAdviceGenerator:
    def generate(
        self,
        region: dict[str, Any],
        risk: RiskAnalysis,
        observations: list[WeatherRiskPoint],
    ) -> AdviceResult:
        recommendations = {
            "low": "기상 정보를 주기적으로 확인하고 가까운 대피소 위치를 미리 확인하세요.",
            "moderate": "비탈면과 하천 주변 접근을 줄이고 재난 알림을 켜 두세요.",
            "high": "외출을 자제하고 대피 준비물을 챙긴 뒤 지자체 안내에 따라 이동하세요.",
            "critical": "즉시 위험 지역을 벗어나 가장 가까운 대피소로 이동하고 공식 안내를 따르세요.",
        }
        recommendation = recommendations[risk.level]
        shelters = region.get("shelters", [])
        if shelters:
            recommendation += (
                f" 등록된 대피소 중 {shelters[0]['name']}"
                f"({shelters[0]['address']})의 위치를 확인하세요."
            )
        return AdviceResult(
            cascading_disasters=list(region.get("cascading_disasters", [])),
            action_recommendation=recommendation,
            generated_by="fallback",
        )


class LangChainAdviceGenerator:
    """LangChain을 통해 OpenAI 호환 LLM에서 구조화된 상세 분석을 생성합니다."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = "",
        timeout_seconds: float = 30,
        chain: Any | None = None,
    ) -> None:
        if chain is not None:
            self.chain = chain
            return
        if not api_key or not model_name:
            raise ValueError("LLM_API_KEY와 LLM_MODEL이 모두 필요합니다.")

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        llm_options: dict[str, Any] = {
            "api_key": api_key,
            "model": model_name,
            "temperature": 0,
            "timeout": timeout_seconds,
        }
        if base_url:
            llm_options["base_url"] = base_url

        llm = ChatOpenAI(**llm_options)
        structured_llm = llm.with_structured_output(LlmAdviceOutput)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 대한민국 재난 안전 분석가입니다. 제공된 데이터만 근거로 "
                    "연쇄 재난과 행동을 안내하세요. 위험도를 과장하지 말고, 대피소는 "
                    "입력에 있는 시설만 언급하세요. 과거 재난을 현재 발생 사실처럼 "
                    "표현하지 마세요. 공식 지자체·기상청 안내를 우선하도록 작성하세요.",
                ),
                ("human", "분석 입력 JSON:\n{context_json}"),
            ]
        )
        self.chain = prompt | structured_llm

    def generate(
        self,
        region: dict[str, Any],
        risk: RiskAnalysis,
        observations: list[WeatherRiskPoint],
    ) -> AdviceResult:
        context = {
            "region": {"id": region["id"], "name": region["name"]},
            "risk": risk.model_dump(),
            "weather_observations_and_risk": [
                item.model_dump() for item in observations
            ],
            "past_disaster_history": region.get("disaster_history", []),
            "verified_shelters": region.get("shelters", []),
        }
        try:
            raw_result = self.chain.invoke(
                {"context_json": json.dumps(context, ensure_ascii=False)}
            )
            result = (
                raw_result
                if isinstance(raw_result, LlmAdviceOutput)
                else LlmAdviceOutput.model_validate(raw_result)
            )
        except Exception as error:
            raise AdviceGenerationError(f"LLM 상세 분석 생성 실패: {error}") from error

        return AdviceResult(
            cascading_disasters=[item.model_dump() for item in result.cascading_disasters],
            action_recommendation=result.action_recommendation,
            generated_by="llm",
        )


class ResilientAdviceGenerator:
    """LLM 장애 시 규칙 기반 결과로 안전하게 대체합니다."""

    def __init__(self, primary: AdviceGenerator, fallback: AdviceGenerator) -> None:
        self.primary = primary
        self.fallback = fallback

    def generate(
        self,
        region: dict[str, Any],
        risk: RiskAnalysis,
        observations: list[WeatherRiskPoint],
    ) -> AdviceResult:
        try:
            return self.primary.generate(region, risk, observations)
        except AdviceGenerationError as error:
            fallback_result = self.fallback.generate(region, risk, observations)
            return AdviceResult(
                cascading_disasters=fallback_result.cascading_disasters,
                action_recommendation=fallback_result.action_recommendation,
                generated_by="fallback",
                error=str(error),
            )
