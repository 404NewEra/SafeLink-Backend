import csv

from repositories.shelter_repository import ShelterRepository


def test_shelter_rag_retrieves_only_selected_district(tmp_path) -> None:
    csv_path = tmp_path / "shelters.csv"
    headers = [
        "연번", "대피장소내용", "대피소주소", "대피가능인원수",
        "담당자전화번호", "대피장소응급시설내용", "비고",
    ]
    with csv_path.open("w", encoding="cp949", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerow(
            {
                "연번": "1",
                "대피장소내용": "관악 대피소",
                "대피소주소": "관악구 신림로 1",
                "대피가능인원수": "50",
                "담당자전화번호": "02-123-4567",
                "대피장소응급시설내용": "구호 세트",
                "비고": "관악산",
            }
        )
        writer.writerow(
            {
                "연번": "2",
                "대피장소내용": "서초 대피소",
                "대피소주소": "서초구 방배로 1",
                "대피가능인원수": "100",
            }
        )

    result = ShelterRepository(str(csv_path), top_k=5).retrieve("서울특별시 관악구")

    assert len(result) == 1
    assert result[0]["name"] == "관악 대피소"
    assert result[0]["capacity"] == 50
    assert result[0]["latitude"] is None
