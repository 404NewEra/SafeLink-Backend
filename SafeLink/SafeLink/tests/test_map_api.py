import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import app  # noqa: E402
from repositories.region_repository import get_region_repository  # noqa: E402
from services.advisory import FallbackAdviceGenerator  # noqa: E402
from services.map_service import MapService, get_map_service  # noqa: E402
from services.risk_model import FallbackRiskPredictor  # noqa: E402
from services.weather_client import SampleWeatherClient  # noqa: E402

app.dependency_overrides[get_map_service] = lambda: MapService(
    repository=get_region_repository(),
    risk_predictor=FallbackRiskPredictor(),
    advice_generator=FallbackAdviceGenerator(),
    weather_client=SampleWeatherClient(),
)

client = TestClient(app)


def test_unused_root_and_health_routes_are_removed() -> None:
    assert client.get("/").status_code == 404
    assert client.get("/health").status_code == 404


def test_map_returns_geojson_for_maplibre() -> None:
    response = client.get("/map")

    assert response.status_code == 200
    body = response.json()
    assert body["geojson"]["type"] == "FeatureCollection"
    assert body["geojson"]["features"]
    assert body["geojson"]["features"][0]["properties"]["risk_color"].startswith("#")
    assert body["geojson"]["features"][0]["id"] == "11620"
    assert len(body["geojson"]["features"]) == 1
    assert body["geojson"]["features"][0]["properties"]["name"].startswith("서울")
    assert len(body["geojson"]["features"][0]["properties"]["observations"]) == 7
    assert body["weather_source"] == "sample"


def test_region_can_be_selected_by_id() -> None:
    response = client.get("/map/11620")

    assert response.status_code == 200
    body = response.json()
    assert body["region"]["id"] == "11620"
    model_input = body["risk"]["model_input"]
    assert list(model_input) == [
        "landslide_map_value",
        "rain_15m",
        "rain_60m",
        "rain_3h",
        "rain_6h",
        "rain_24h",
        "slope",
        "elevation",
    ]
    assert model_input["landslide_map_value"] in range(1, 6)
    assert model_input["rain_60m"] >= 0
    assert model_input["slope"] >= 0
    assert body["weather_summary"]["observation_count"] == 7
    assert len(body["observations"]) == 7
    assert isinstance(body["disaster_history"], list)
    assert isinstance(body["cascading_disasters"], list)
    assert body["action_recommendation"]
    assert isinstance(body["shelters"], list)


def test_region_can_be_searched_by_name() -> None:
    response = client.get("/map/관악")

    assert response.status_code == 200
    assert response.json()["region"]["id"] == "11620"


def test_full_name_and_legacy_id_are_normalized_to_official_code() -> None:
    for value in ("서울특별시 관악구", "서울 관악구", "seoul-gwanak"):
        response = client.get(f"/map/{value}")
        assert response.status_code == 200
        assert response.json()["region"]["id"] == "11620"


def test_non_seoul_region_is_not_exposed() -> None:
    assert client.get("/map/춘천").status_code == 404


def test_unknown_region_returns_404() -> None:
    response = client.get("/map/없는지역")

    assert response.status_code == 404
