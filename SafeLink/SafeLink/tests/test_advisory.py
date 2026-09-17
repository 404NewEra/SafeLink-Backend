import json

from models.schemas import RandomForestInput, RiskAnalysis, WeatherRiskPoint
from services.advisory import (
    FallbackAdviceGenerator,
    LangChainAdviceGenerator,
    ResilientAdviceGenerator,
)


def _risk() -> RiskAnalysis:
    return RiskAnalysis(
        overall_score=78,
        level="critical",
        label="매우 높음",
        color="#E74C3C",
        model_input=RandomForestInput(
            landslide_map_value=1,
            rain_15m=10.5,
            rain_60m=42,
            rain_3h=70,
            rain_6h=100,
            rain_24h=150,
            slope=31.2,
            elevation=250,
        ),
        source="fallback",
    )


def _observations() -> list[WeatherRiskPoint]:
    return [
        WeatherRiskPoint(
            observed_at="2026-09-17T14:00:00+09:00",
            temperature_c=23,
            dew_point_c=20,
            humidity_percent=90,
            risk=_risk(),
        )
    ]


def _region() -> dict:
    return {
        "id": "gangwon-chuncheon",
        "name": "강원특별자치도 춘천시",
        "disaster_history": [
            {"date": "2020-08", "type": "산사태", "description": "토사 유출"}
        ],
        "cascading_disasters": [
            {"type": "도로 통제", "possibility": "높음", "description": "토사 유출"}
        ],
        "shelters": [
            {
                "id": "s1",
                "name": "시민체육센터",
                "address": "강원특별자치도 춘천시",
                "latitude": 37.8,
                "longitude": 127.7,
            }
        ],
    }


class FakeChain:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.context: dict | None = None

    def invoke(self, values: dict) -> dict:
        self.context = json.loads(values["context_json"])
        if self.should_fail:
            raise RuntimeError("temporary LLM failure")
        return {
            "cascading_disasters": [
                {
                    "type": "산사태 후 도로 통제",
                    "possibility": "매우 높음",
                    "description": "강수량과 경사도가 모두 높습니다.",
                }
            ],
            "action_recommendation": "공식 안내에 따라 시민체육센터로 이동하세요.",
        }


def test_langchain_generator_receives_risk_history_and_verified_shelters() -> None:
    chain = FakeChain()
    generator = LangChainAdviceGenerator("", "", chain=chain)

    result = generator.generate(_region(), _risk(), _observations())

    assert result.generated_by == "llm"
    assert result.cascading_disasters[0]["type"] == "산사태 후 도로 통제"
    assert chain.context is not None
    assert chain.context["risk"]["model_input"]["rain_60m"] == 42
    assert chain.context["risk"]["model_input"]["slope"] == 31.2
    assert chain.context["past_disaster_history"][0]["type"] == "산사태"
    assert chain.context["verified_shelters"][0]["name"] == "시민체육센터"


def test_llm_failure_falls_back_without_losing_shelter_guidance() -> None:
    generator = ResilientAdviceGenerator(
        primary=LangChainAdviceGenerator("", "", chain=FakeChain(should_fail=True)),
        fallback=FallbackAdviceGenerator(),
    )

    result = generator.generate(_region(), _risk(), _observations())

    assert result.generated_by == "fallback"
    assert result.error is not None
    assert "시민체육센터" in result.action_recommendation
