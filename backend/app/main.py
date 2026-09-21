from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.adapters.llm import build_llm_adapter
from app.adapters.ocr import build_ocr_adapter
from app.adapters.rag import build_rag_adapter
from app.api.routes import router
from app.config import get_settings
from app.logging_setup import configure_logging
from app.services.pipeline import DocumentPipeline


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    ocr = build_ocr_adapter(settings)
    rag = build_rag_adapter(settings)
    llm = build_llm_adapter(settings)
    await rag.ensure_indexed()
    application.state.pipeline = DocumentPipeline(settings=settings, ocr=ocr, rag=rag, llm=llm)
    yield
    application.state.pipeline = None


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="Мосгортранс — обработка обращений",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
