import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@dataclass(frozen=True)
class Settings:
    cors_origins: list[str] = field(default_factory=_cors_origins)
    random_forest_model_path: str = field(
        default_factory=lambda: os.getenv("RANDOM_FOREST_MODEL_PATH", "")
    )
    kma_auth_key: str = field(
        default_factory=lambda: os.getenv(
            "KMA_AUTH_KEY", os.getenv("KMA_SERVICE_KEY", "")
        )
    )
    kma_api_url: str = field(
        default_factory=lambda: os.getenv(
            "KMA_API_URL",
            "https://apihub.kma.go.kr/api/typ01/url/sfc_nc_var.php",
        )
    )
    kma_api_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("KMA_API_TIMEOUT_SECONDS", "10"))
    )
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    llm_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    )
    landslide_history_csv_path: str = field(
        default_factory=lambda: os.getenv("LANDSLIDE_HISTORY_CSV_PATH", "")
    )
    landslide_risk_raster_path: str = field(
        default_factory=lambda: os.getenv("LANDSLIDE_RISK_RASTER_PATH", "")
    )
    dem_raster_path: str = field(
        default_factory=lambda: os.getenv("DEM_RASTER_PATH", "")
    )
    shelter_csv_path: str = field(
        default_factory=lambda: os.getenv("SHELTER_CSV_PATH", "")
    )
    shelter_rag_top_k: int = field(
        default_factory=lambda: int(os.getenv("SHELTER_RAG_TOP_K", "5"))
    )
    mvp_region_prefix: str = field(
        default_factory=lambda: os.getenv("MVP_REGION_PREFIX", "서울특별시")
    )


settings = Settings()
