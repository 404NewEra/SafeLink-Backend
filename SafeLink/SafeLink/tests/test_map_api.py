import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import app  # noqa: E402

client = TestClient(app)


def test_root_returns_api_guide_instead_of_404() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_map_returns_geojson_for_maplibre() -> None:
    response = client.get("/map")

    assert response.status_code == 200
    body = response.json()
    assert body["geojson"]["type"] == "FeatureCollection"
    assert body["geojson"]["features"]
    assert body["geojson"]["features"][0]["properties"]["risk_color"].startswith("#")


def test_region_can_be_selected_by_id() -> None:
    response = client.get("/map/seoul-gwanak")

    assert response.status_code == 200
    body = response.json()
    assert body["region"]["id"] == "seoul-gwanak"
    assert body["risk"]["rainfall_mm"] >= 0
    assert body["shelters"]


def test_region_can_be_searched_by_name() -> None:
    response = client.get("/map/춘천")

    assert response.status_code == 200
    assert response.json()["region"]["id"] == "gangwon-chuncheon"


def test_unknown_region_returns_404() -> None:
    response = client.get("/map/없는지역")

    assert response.status_code == 404
