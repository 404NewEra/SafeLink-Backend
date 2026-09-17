import csv
from functools import lru_cache
from pathlib import Path

from core.config import settings


class ShelterRepository:
    """서울시 CSV에서 선택 지역과 일치하는 대피소 문서를 검색합니다."""

    def __init__(self, csv_path: str, top_k: int = 5) -> None:
        self.csv_path = Path(csv_path) if csv_path else None
        self.top_k = max(1, top_k)

    def retrieve(self, region_name: str) -> list[dict]:
        if self.csv_path is None or not self.csv_path.is_file():
            return []
        parts = region_name.split()
        district = parts[1] if len(parts) > 1 else ""
        matches: list[dict] = []
        with self.csv_path.open(encoding="cp949", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                address = row.get("대피소주소", "").strip()
                if district and district not in address.split():
                    continue
                capacity_text = row.get("대피가능인원수", "").strip().replace(",", "")
                try:
                    capacity = int(capacity_text)
                except ValueError:
                    capacity = None
                sequence = row.get("연번", "").strip() or str(len(matches) + 1)
                matches.append(
                    {
                        "id": f"seoul-landslide-shelter-{sequence}",
                        "name": row.get("대피장소내용", "").strip(),
                        "address": (
                            address
                            if address.startswith("서울")
                            else f"서울특별시 {address}"
                        ),
                        "latitude": None,
                        "longitude": None,
                        "capacity": capacity,
                        "phone": row.get("담당자전화번호", "").strip() or None,
                        "emergency_facilities": (
                            row.get("대피장소응급시설내용", "").strip() or None
                        ),
                        "note": row.get("비고", "").strip() or None,
                    }
                )
                if len(matches) >= self.top_k:
                    break
        return matches


@lru_cache
def get_shelter_repository() -> ShelterRepository:
    return ShelterRepository(settings.shelter_csv_path, settings.shelter_rag_top_k)
