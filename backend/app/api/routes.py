from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.api.schemas import ExportDocxRequest, HealthResponse, ProcessDocumentResponse
from app.config import Settings, get_settings
from app.services.docx_export import build_docx
from app.services.pipeline import DocumentPipeline

router = APIRouter(prefix="/api/v1")


def get_pipeline(request: Request) -> DocumentPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Конвейер обработки ещё не готов")
    return pipeline


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        ocr_provider=settings.ocr_provider,
        llm_provider=settings.llm_provider,
        rag_provider=settings.rag_provider,
    )


@router.post("/process-document", response_model=ProcessDocumentResponse)
async def process_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    pipeline: DocumentPipeline = Depends(get_pipeline),
) -> ProcessDocumentResponse:
    filename = file.filename or "upload.bin"
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Допустимые форматы: {', '.join(sorted(settings.allowed_suffixes))}",
        )
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Файл пуст")
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Файл превышает лимит 20 МБ")
    try:
        result = await pipeline.process(payload, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки документа: {exc}") from exc
    return ProcessDocumentResponse(
        request_id=result.request_id,
        source_text=result.source_text,
        answer_text=result.answer_text,
        ocr_method=result.ocr_method,
        rag_status=result.rag_status,
        llm_provider=result.llm_provider,
        warnings=result.warnings,
        rag_sources=result.rag_sources,
    )


@router.post("/export-docx")
async def export_docx(body: ExportDocxRequest) -> Response:
    if not body.answer_text.strip():
        raise HTTPException(status_code=400, detail="Текст ответа пуст")
    content = build_docx(body.answer_text, body.request_id)
    headers = {"Content-Disposition": 'attachment; filename="mosgortrans-reply.docx"'}
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers,
    )
