from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

from app.adapters.llm import LLMAdapter, LLMResult
from app.adapters.ocr import OCRAdapter, OCRResult
from app.adapters.rag import RAGAdapter, RAGResult
from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    request_id: str
    source_text: str
    answer_text: str
    ocr_method: str
    rag_status: str
    llm_provider: str
    warnings: list[str]
    rag_sources: list[str]


class DocumentPipeline:
    def __init__(
        self,
        settings: Settings,
        ocr: OCRAdapter,
        rag: RAGAdapter,
        llm: LLMAdapter,
    ) -> None:
        self.settings = settings
        self.ocr = ocr
        self.rag = rag
        self.llm = llm

    async def process(self, payload: bytes, filename: str) -> PipelineResult:
        request_id = str(uuid.uuid4())
        warnings: list[str] = []
        started = time.perf_counter()
        logger.info(
            "pipeline_started",
            extra={"event": "pipeline_started", "request_id": request_id, "upload_name": filename},
        )

        ocr_result = await self._run_ocr(payload, filename, request_id)
        if not ocr_result.success or not ocr_result.text.strip():
            raise ValueError(ocr_result.error or "Не удалось извлечь текст из файла")

        rag_result = await self._run_rag(ocr_result.text, request_id)
        if rag_result.status != "ok":
            warnings.append("Поиск по базе НПА выполнен в ограниченном режиме либо не дал результатов.")

        llm_result = await self._run_llm(ocr_result.text, rag_result.context_text, request_id)
        answer = llm_result.text
        if not llm_result.success:
            warnings.append(
                "Генерация ответа моделью недоступна. Показан распознанный текст обращения; "
                f"причина: {llm_result.error or 'неизвестная ошибка'}."
            )
            answer = _fallback_answer(ocr_result.text, llm_result.error)

        logger.info(
            "pipeline_completed",
            extra={
                "event": "pipeline_completed",
                "request_id": request_id,
                "success": True,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "upload_name": filename,
            },
        )
        return PipelineResult(
            request_id=request_id,
            source_text=ocr_result.text,
            answer_text=answer,
            ocr_method=ocr_result.method,
            rag_status=rag_result.status,
            llm_provider=llm_result.provider,
            warnings=warnings,
            rag_sources=[hit.source for hit in rag_result.hits],
        )

    async def _run_ocr(self, payload: bytes, filename: str, request_id: str) -> OCRResult:
        started = time.perf_counter()
        logger.info("ocr_started", extra={"event": "ocr_started", "request_id": request_id, "upload_name": filename})
        try:
            result = await asyncio.wait_for(
                self.ocr.extract_text(payload, filename),
                timeout=self.settings.ocr_timeout_seconds,
            )
            logger.info(
                "ocr_completed",
                extra={
                    "event": "ocr_completed",
                    "request_id": request_id,
                    "success": result.success,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "ocr_chars": len(result.text),
                    "provider": result.method,
                    "error": result.error,
                },
            )
            return result
        except asyncio.TimeoutError:
            logger.error(
                "ocr_timeout",
                extra={
                    "event": "ocr_timeout",
                    "request_id": request_id,
                    "success": False,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "error": "OCR timeout",
                },
            )
            return OCRResult("", "timeout", False, "Превышено время ожидания OCR (30 сек)")
        except Exception as exc:
            logger.exception(
                "ocr_failed",
                extra={"event": "ocr_failed", "request_id": request_id, "success": False, "error": str(exc)},
            )
            return OCRResult("", "error", False, str(exc))

    async def _run_rag(self, query: str, request_id: str) -> RAGResult:
        started = time.perf_counter()
        logger.info("rag_started", extra={"event": "rag_started", "request_id": request_id})
        try:
            result = await asyncio.wait_for(
                self.rag.retrieve(query, self.settings.rag_top_k),
                timeout=self.settings.rag_timeout_seconds,
            )
            logger.info(
                "rag_completed",
                extra={
                    "event": "rag_completed",
                    "request_id": request_id,
                    "success": True,
                    "status": result.status,
                    "rag_hits": len(result.hits),
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "error": result.error,
                },
            )
            return result
        except Exception as exc:
            logger.warning(
                "rag_failed",
                extra={
                    "event": "rag_failed",
                    "request_id": request_id,
                    "success": False,
                    "status": "failed",
                    "error": str(exc),
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            return RAGResult(hits=[], status="failed", error=str(exc))

    async def _run_llm(self, inquiry_text: str, rag_context: str, request_id: str) -> LLMResult:
        started = time.perf_counter()
        logger.info("llm_started", extra={"event": "llm_started", "request_id": request_id})
        try:
            result = await asyncio.wait_for(
                self.llm.generate_reply(inquiry_text, rag_context),
                timeout=self.settings.llm_timeout_seconds,
            )
            logger.info(
                "llm_completed",
                extra={
                    "event": "llm_completed",
                    "request_id": request_id,
                    "success": result.success,
                    "provider": result.provider,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "error": result.error,
                },
            )
            return result
        except asyncio.TimeoutError:
            logger.error(
                "llm_timeout",
                extra={
                    "event": "llm_timeout",
                    "request_id": request_id,
                    "success": False,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "error": "LLM timeout",
                },
            )
            return LLMResult("", False, "timeout", "Превышено время ожидания LLM (45 сек)")
        except Exception as exc:
            logger.exception(
                "llm_failed",
                extra={"event": "llm_failed", "request_id": request_id, "success": False, "error": str(exc)},
            )
            return LLMResult("", False, "error", str(exc))


def _fallback_answer(source_text: str, error: str | None) -> str:
    return (
        "Проект официального ответа не сформирован: языковая модель недоступна "
        f"({error or 'ошибка генерации'}).\n\n"
        "Ниже приведён распознанный текст обращения для ручной подготовки ответа:\n\n"
        f"{source_text}"
    )
