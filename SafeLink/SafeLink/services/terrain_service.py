from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window


class TerrainDataError(RuntimeError):
    """지형 래스터를 읽거나 유효한 값을 찾지 못했습니다."""


@dataclass(frozen=True)
class TerrainFeatures:
    landslide_map_value: int
    slope: float
    elevation: float
    landslide_sample_distance_m: float
    source: str = "forest_service_raster_and_dem"


class TerrainService:
    def __init__(self, landslide_raster_path: str, dem_raster_path: str) -> None:
        self.landslide_raster_path = Path(landslide_raster_path)
        self.dem_raster_path = Path(dem_raster_path)

    @staticmethod
    def _pixel_for_lon_lat(dataset, longitude: float, latitude: float) -> tuple[int, int]:
        if dataset.crs is None:
            raise TerrainDataError("래스터 CRS가 없습니다.")
        xs, ys = transform("EPSG:4326", dataset.crs, [longitude], [latitude])
        return dataset.index(xs[0], ys[0])

    def _landslide_value(self, longitude: float, latitude: float) -> tuple[int, float]:
        if not self.landslide_raster_path.is_file():
            raise TerrainDataError(f"산사태 위험지도 파일이 없습니다: {self.landslide_raster_path}")

        with rasterio.open(self.landslide_raster_path) as dataset:
            row, column = self._pixel_for_lon_lat(dataset, longitude, latitude)
            for radius in (0, 10, 25, 50, 100, 250, 500):
                size = radius * 2 + 1
                values = dataset.read(
                    1,
                    window=Window(column - radius, row - radius, size, size),
                    boundless=True,
                    fill_value=dataset.nodata,
                )
                valid_rows, valid_columns = np.where(np.isin(values, [1, 2, 3, 4, 5]))
                if not len(valid_rows):
                    continue
                distances = np.hypot(valid_rows - radius, valid_columns - radius)
                nearest = int(np.argmin(distances))
                pixel_distance = float(distances[nearest])
                pixel_size = max(abs(dataset.transform.a), abs(dataset.transform.e))
                return int(values[valid_rows[nearest], valid_columns[nearest]]), round(
                    pixel_distance * pixel_size, 2
                )
        raise TerrainDataError("좌표 주변 5km 이내에 유효한 산사태 위험등급이 없습니다.")

    def _dem_features(self, longitude: float, latitude: float) -> tuple[float, float]:
        if not self.dem_raster_path.is_file():
            raise TerrainDataError(f"DEM 파일이 없습니다: {self.dem_raster_path}")

        with rasterio.open(self.dem_raster_path) as dataset:
            row, column = self._pixel_for_lon_lat(dataset, longitude, latitude)
            values = dataset.read(
                1,
                window=Window(column - 1, row - 1, 3, 3),
                boundless=True,
                fill_value=dataset.nodata,
            ).astype(float)
            if dataset.nodata is not None:
                values[values == dataset.nodata] = np.nan
            if values.shape != (3, 3) or np.isnan(values).any():
                raise TerrainDataError("DEM 좌표 주변 3x3 격자에 유효하지 않은 값이 있습니다.")

            y_gradient, x_gradient = np.gradient(
                values, abs(dataset.transform.e), abs(dataset.transform.a)
            )
            slope = np.degrees(
                np.arctan(np.hypot(x_gradient[1, 1], y_gradient[1, 1]))
            )
            return round(float(values[1, 1]), 3), round(float(slope), 3)

    def get_features(self, center: list[float]) -> TerrainFeatures:
        longitude, latitude = center
        grade, distance = self._landslide_value(longitude, latitude)
        elevation, slope = self._dem_features(longitude, latitude)
        return TerrainFeatures(
            landslide_map_value=grade,
            slope=slope,
            elevation=elevation,
            landslide_sample_distance_m=distance,
        )
