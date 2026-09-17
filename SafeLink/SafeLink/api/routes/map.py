from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from models.schemas import MapResponse, RegionDetailResponse
from services.map_service import MapService, get_map_service

router = APIRouter(prefix="/map", tags=["map"])


@router.get(
    "",
    response_model=MapResponse,
    summary="지도 위험도 분포 조회",
    description="MapLibre GeoJSON source로 사용할 지역별 종합 위험도와 색상을 반환합니다.",
)
def get_map(
    service: Annotated[MapService, Depends(get_map_service)],
) -> MapResponse:
    return service.get_map()


@router.get(
    "/{region_id}",
    response_model=RegionDetailResponse,
    summary="지역 선택 또는 검색 결과 조회",
    description="지역 ID, 지역명 또는 별칭으로 지역을 찾아 위험 분석과 행동 추천을 반환합니다.",
    responses={status.HTTP_404_NOT_FOUND: {"description": "지역을 찾을 수 없음"}},
)
def get_region_detail(
    region_id: Annotated[
        str,
        Path(
            min_length=1,
            max_length=100,
            description="5자리 시군구 코드 또는 검색할 지역명 (예: 11620, 관악구)",
        ),
    ],
    service: Annotated[MapService, Depends(get_map_service)],
) -> RegionDetailResponse:
    detail = service.get_region_detail(region_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{region_id}'에 해당하는 지역을 찾을 수 없습니다.",
        )
    return detail
