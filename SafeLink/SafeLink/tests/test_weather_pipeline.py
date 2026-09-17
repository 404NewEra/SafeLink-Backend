from datetime import datetime

import httpx

from services.risk_model import FallbackRiskPredictor, RandomForestRiskPredictor
from services.weather_client import KST, KmaWeatherClient, WeatherObservation


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_kma_apihub_response_is_parsed_and_request_is_bounded_to_60_minutes(
    monkeypatch,
) -> None:
    captured_params: dict = {}
    response_text = """
# TM TA TD HM RN15 RN60 RN03 RN06 RNDAY
202609171130 21.0 18.0 82 1.5 4.0 8.0 12.0 20.0
202609171140 21.2 18.1 81 2.0 5.0 9.0 13.0 22.0
"""

    def fake_get(url: str, *, params: dict, timeout: float) -> FakeResponse:
        captured_params.update(params)
        assert url == "https://example.test/weather"
        assert timeout == 5
        return FakeResponse(response_text)

    monkeypatch.setattr(httpx, "get", fake_get)
    observations = KmaWeatherClient(
        "test-key", "https://example.test/weather", 5
    ).get_recent_observations(
        {"center": [126.9516, 37.4784]},
        now=datetime(2026, 9, 17, 12, 47, tzinfo=KST),
    )

    assert captured_params["tm1"] == "202609171140"
    assert captured_params["tm2"] == "202609171240"
    assert captured_params["itv"] == 10
    assert captured_params["authKey"] == "test-key"
    assert captured_params["obs"] == (
        "ta,td,hm,rn_15m,rn_60m,rn_03h,rn_06h,rn_day"
    )
    assert len(observations) == 2
    assert observations[1].rain_15m == 2
    assert observations[1].rain_24h == 22


def test_exact_kma_rainfall_fields_are_used_as_random_forest_input(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: FakeResponse(
            "202609171400 23 20 90 10 40 70 100 150\n"
        ),
    )
    weather = KmaWeatherClient(
        "test-key", "https://example.test/weather"
    ).get_recent_observations(
        {"center": [127.0, 37.5]},
        now=datetime(2026, 9, 17, 14, 0, tzinfo=KST),
    )[0]
    risk = FallbackRiskPredictor().predict(
        {"slope": 30, "elevation": 250, "landslide_map_value": 1}, weather
    )

    assert risk.model_input.rain_15m == 10
    assert risk.model_input.rain_60m == 40
    assert risk.model_input.rain_3h == 70
    assert risk.model_input.rain_6h == 100
    assert risk.model_input.rain_24h == 150
    assert risk.overall_score > 50


class FakeRandomForest:
    classes_ = [0, 1]

    def predict_proba(self, vector):
        assert len(vector[0]) == 8
        return [[0.27, 0.73]]


def test_random_forest_positive_probability_becomes_landslide_risk() -> None:
    predictor = RandomForestRiskPredictor("unused.joblib")
    predictor.model = FakeRandomForest()
    observation = WeatherObservation(
        observed_at=datetime(2026, 9, 17, 14, 0, tzinfo=KST),
        temperature_c=23,
        dew_point_c=20,
        humidity_percent=90,
        rain_15m=20,
        rain_60m=50,
        rain_3h=90,
        rain_6h=150,
        rain_24h=300,
        source="test",
    )
    risk = predictor.predict(
        {"slope": 30, "elevation": 250, "landslide_map_value": 1}, observation
    )

    assert risk.landslide_probability == 0.73
    assert risk.landslide_predicted is True
    assert risk.landslide_risk_score == 73
    assert risk.heavy_rain_risk_score == 100
    assert risk.cascade_risk_score == 85.44
    assert risk.overall_score == 83.59
    assert risk.weights.model_dump() == {
        "landslide": 0.5,
        "heavy_rain": 0.3,
        "cascade": 0.2,
    }
