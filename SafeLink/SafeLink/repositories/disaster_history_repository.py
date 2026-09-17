import csv
from functools import lru_cache
from pathlib import Path

from core.config import settings


class DisasterHistoryRepository:
    """산림청 산사태 발생 이력 CSV를 지역별로 조회합니다."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path) if csv_path else None

    def find_by_region(self, region_name: str) -> list[dict[str, str]]:
        if self.csv_path is None or not self.csv_path.is_file():
            return []

        parts = region_name.split()
        province = parts[0] if parts else ""
        district = parts[1] if len(parts) > 1 else ""
        histories: list[dict[str, str]] = []
        with self.csv_path.open(encoding="cp949", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                if row.get("상세주소_시도", "").strip() != province:
                    continue
                if district and row.get("상세주소_시군구", "").strip() != district:
                    continue

                locality = " ".join(
                    value.strip()
                    for key in ("상세주소_읍면동", "상세주소_리")
                    if (value := row.get(key, "")) and value.strip()
                )
                damage = row.get("피해물량(ha)", "").strip()
                details = [part for part in (locality, f"피해면적 {damage}ha" if damage else "") if part]
                histories.append(
                    {
                        "date": row.get("연도", "").strip(),
                        "type": row.get("시설구분_등급", "").strip() or "산사태",
                        "description": ", ".join(details) or "산림청 산사태 발생 이력",
                    }
                )
        return histories


@lru_cache
def get_disaster_history_repository() -> DisasterHistoryRepository:
    return DisasterHistoryRepository(settings.landslide_history_csv_path)
