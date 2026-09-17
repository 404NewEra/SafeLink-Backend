import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config import settings

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "regions.json"


def _normalize(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value).casefold()


class RegionRepository:
    def __init__(self, data_path: Path = DATA_PATH) -> None:
        with data_path.open(encoding="utf-8") as file:
            regions: list[dict[str, Any]] = json.load(file)
        self._regions = [
            region
            for region in regions
            if region["name"].startswith(settings.mvp_region_prefix)
        ]

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
