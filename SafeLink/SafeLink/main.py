from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.map import router as map_router
from core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="SafeLink API",
        version="0.1.0",
        description="MapLibre 지도용 재난 위험도 및 지역 상세 정보 API",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(map_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    # IntelliJ에서 이 파일을 직접 실행하면 개발 서버가 시작됩니다.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
