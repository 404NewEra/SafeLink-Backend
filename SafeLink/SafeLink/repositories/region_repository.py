import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config import settings

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "seoul_municipalities_source.geojson"
)

CURRENT_DISTRICT_CODES = {
    "Jongno-gu": "11110",
    "Jung-gu": "11140",
    "Yongsan-gu": "11170",
    "Seongdong-gu": "11200",
    "Gwangjin-gu": "11215",
    "Dongdaemun-gu": "11230",
    "Jungnang-gu": "11260",
    "Seongbuk-gu": "11290",
    "Gangbuk-gu": "11305",
    "Dobong-gu": "11320",
    "Nowon-gu": "11350",
    "Eunpyeong-gu": "11380",
    "Seodaemun-gu": "11410",
    "Mapo-gu": "11440",
    "Yangcheon-gu": "11470",
    "Gangseo-gu": "11500",
    "Guro-gu": "11530",
    "Geumcheon-gu": "11545",
    "Yeongdeungpo-gu": "11560",
    "Dongjak-gu": "11590",
    "Gwanak-gu": "11620",
    "Seocho-gu": "11650",
    "Gangnam-gu": "11680",
    "Songpa-gu": "11710",
    "Gangdong-gu": "11740",
}


def _normalize(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value).casefold()


class RegionRepository:
    def __init__(self, data_path: Path = DATA_PATH) -> None:
        with data_path.open(encoding="utf-8") as file:
            geojson: dict[str, Any] = json.load(file)
        regions = [self._to_region(feature) for feature in geojson["features"]]
        self._regions = [
            region
            for region in regions
            if region["name"].startswith(settings.mvp_region_prefix)
        ]

    @staticmethod
    def _to_region(feature: dict[str, Any]) -> dict[str, Any]:
        properties = feature["properties"]
        district = properties["name"]
        english_name = properties["name_eng"]
        code = CURRENT_DISTRICT_CODES[english_name]
        outer_ring = feature["geometry"]["coordinates"][0]
        longitude = sum(point[0] for point in outer_ring) / len(outer_ring)
        latitude = sum(point[1] for point in outer_ring) / len(outer_ring)
        short_name = district.removesuffix("구")
        english_short_name = english_name.removesuffix("-gu")
        return {
            "id": code,
            "name": f"서울특별시 {district}",
            "aliases": [
                f"서울 {district}",
                district,
                short_name,
                english_name,
                english_short_name,
                f"seoul-{english_short_name.lower()}",
            ],
            "center": [round(longitude, 7), round(latitude, 7)],
            "geometry": feature["geometry"],
            "sample_rain_60m": 0.0,
            "slope": 0.0,
            "elevation": 0.0,
            "landslide_map_value": 5,
            "terrain_source": "fallback",
            "disaster_history": [],
            "cascading_disasters": [],
            "shelters": [],
        }

    def list_all(self) -> list[dict[str, Any]]:
        return self._regions.copy()

    def find(self, identifier_or_query: str) -> dict[str, Any] | None:
        query = _normalize(identifier_or_query)
        if not query:
            return None

        for region in self._regions:
            exact_candidates = [region["id"], region["name"], *region.get("aliases", [])]
            if any(_normalize(candidate) == query for candidate in exact_candidates):
                return region

        for region in self._regions:
            search_text = [region["name"], *region.get("aliases", [])]
            if any(query in _normalize(candidate) for candidate in search_text):
                return region
        return None


@lru_cache
def get_region_repository() -> RegionRepository:
    return RegionRepository()
