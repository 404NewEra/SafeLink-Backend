import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import httpx

KST = timezone(timedelta(hours=9))
OBSERVATION_CODES = (
    "ta", "td", "hm", "rn_15m", "rn_60m", "rn_03h", "rn_06h", "rn_day"
)


class WeatherClientError(RuntimeError):
    """기상청 요청 또는 응답 처리에 실패했을 때 발생합니다."""


@dataclass(frozen=True)
class WeatherObservation:
    observed_at: datetime
    temperature_c: float | None
    dew_point_c: float | None
    humidity_percent: float | None
    rain_15m: float
    rain_60m: float
    rain_3h: float
    rain_6h: float
    rain_24h: float
    source: str
    rain_24h_method: str = "rn_day (KST 자정 이후 누적값)"


class WeatherClient(Protocol):
    def get_recent_observations(
        self, region: dict[str, Any], now: datetime | None = None
    ) -> list[WeatherObservation]:
        """최근 60분의 10분 간격 관측값을 KST 기준으로 반환합니다."""


def _optional_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number <= -90 else number


def _rainfall(value: str) -> float:
    parsed = _optional_float(value)
    return max(0.0, parsed or 0.0)


def parse_kma_observations(text: str) -> list[WeatherObservation]:
    observations: list[WeatherObservation] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "START" in line or "END" in line:
            continue
        columns = [item for item in re.split(r"[\s,]+", line) if item]
        timestamp_index = next(
            (index for index, item in enumerate(columns) if re.fullmatch(r"\d{12}", item)),
            None,
        )
        if timestamp_index is None:
            continue
        values = columns[timestamp_index + 1 :]
        if len(values) < len(OBSERVATION_CODES):
            continue
        values = values[-len(OBSERVATION_CODES) :]
        by_code = dict(zip(OBSERVATION_CODES, values))
        observations.append(
            WeatherObservation(
                observed_at=datetime.strptime(
                    columns[timestamp_index], "%Y%m%d%H%M"
                ).replace(tzinfo=KST),
                temperature_c=_optional_float(by_code["ta"]),
                dew_point_c=_optional_float(by_code["td"]),
                humidity_percent=_optional_float(by_code["hm"]),
                rain_15m=_rainfall(by_code["rn_15m"]),
                rain_60m=_rainfall(by_code["rn_60m"]),
                rain_3h=_rainfall(by_code["rn_03h"]),
                rain_6h=_rainfall(by_code["rn_06h"]),
                rain_24h=_rainfall(by_code["rn_day"]),
                source="kma_apihub",
            )
        )
    return sorted(observations, key=lambda item: item.observed_at)


class KmaWeatherClient:
    def __init__(self, auth_key: str, api_url: str, timeout_seconds: float = 10) -> None:
        if not auth_key.strip():
            raise ValueError("KMA_AUTH_KEY가 설정되지 않았습니다.")
        self.auth_key = auth_key.strip()
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds

    def get_recent_observations(
        self, region: dict[str, Any], now: datetime | None = None
    ) -> list[WeatherObservation]:
        current = (now or datetime.now(KST)).astimezone(KST).replace(second=0, microsecond=0)
        end = current.replace(minute=(current.minute // 10) * 10)
        start = end - timedelta(minutes=60)
        longitude, latitude = region["center"]
        try:
            response = httpx.get(
                self.api_url,
                params={
                    "tm1": start.strftime("%Y%m%d%H%M"),
                    "tm2": end.strftime("%Y%m%d%H%M"),
                    "lon": longitude,
                    "lat": latitude,
                    "obs": ",".join(OBSERVATION_CODES),
                    "itv": 10,
                    "help": 0,
                    "authKey": self.auth_key,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise WeatherClientError(f"기상청 APIHub 요청 실패: {error}") from error

        observations = parse_kma_observations(response.text)
        if not observations:
            message = response.text.strip()[:200]
            raise WeatherClientError(
                f"기상청 APIHub 응답에 유효한 관측값이 없습니다: {message}"
            )
        return observations


class SampleWeatherClient:
    """인증키가 없거나 외부 호출에 실패할 때 사용하는 개발용 데이터입니다."""

    def get_recent_observations(
        self, region: dict[str, Any], now: datetime | None = None
    ) -> list[WeatherObservation]:
        current = (now or datetime.now(KST)).astimezone(KST).replace(second=0, microsecond=0)
        end = current.replace(minute=(current.minute // 10) * 10)
        base_rain = max(0.0, float(region.get("sample_rain_60m", 0.0)))
        return [
            WeatherObservation(
                observed_at=end - timedelta(minutes=60 - offset),
                temperature_c=22.0,
                dew_point_c=18.0,
                humidity_percent=78.0,
                rain_15m=round(base_rain * 0.25, 3),
                rain_60m=base_rain,
                rain_3h=round(base_rain * 1.5, 3),
                rain_6h=round(base_rain * 2.0, 3),
                rain_24h=round(base_rain * 3.0, 3),
                source="sample",
                rain_24h_method="sample",
            )
            for offset in range(0, 61, 10)
        ]
