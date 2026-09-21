from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"


@dataclass
class LLMResult:
    text: str
    success: bool
    provider: str
    error: str | None = None


class LLMAdapter(ABC):
    @abstractmethod
    async def generate_reply(self, inquiry_text: str, rag_context: str) -> LLMResult:
        raise NotImplementedError


def load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "Вы помощник по подготовке официальных ответов ГУП «Мосгортранс»."


class StubLLMAdapter(LLMAdapter):
    async def generate_reply(self, inquiry_text: str, rag_context: str) -> LLMResult:
        excerpt = inquiry_text.strip()[:900] or "текст обращения не извлечён"
        sources = rag_context.strip()[:1200] or "релевантные фрагменты НПА не найдены"
        draft = (
            "ГУП «Мосгортранс»\n\n"
            "Уважаемый заявитель!\n\n"
            "Ваше обращение рассмотрено. Ниже приведён проект ответа, сформированный "
            "в резервном режиме (локальная LLM недоступна). Перед отправкой документ "
            "необходимо проверить сотрудником.\n\n"
            "Краткое содержание обращения:\n"
            f"{excerpt}\n\n"
            "Ориентиры по нормативным материалам:\n"
            f"{sources}\n\n"
            "По существу сообщаем, что организация пассажирских перевозок, рассмотрение "
            "обращений граждан и предоставление информации осуществляются в соответствии "
            "с законодательством РФ и внутренними регламентами предприятия. При необходимости "
            "уточняющих сведений ответ может быть дополнен после проверки фактов.\n\n"
            "С уважением,\n"
            "ГУП «Мосгортранс»\n"
        )
        return LLMResult(text=draft, success=True, provider="stub")


class VLLMLLMAdapter(LLMAdapter):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.system_prompt = load_system_prompt()

    async def generate_reply(self, inquiry_text: str, rag_context: str) -> LLMResult:
        user_prompt = (
            "Подготовьте проект официального ответа на обращение.\n\n"
            "=== Текст обращения ===\n"
            f"{inquiry_text.strip()}\n\n"
            "=== Контекст из базы НПА (используйте только подтверждённые фрагменты) ===\n"
            f"{rag_context.strip() or 'Контекст не найден. Не выдумывайте нормы права.'}\n"
        )
        payload = {
            "model": self.settings.vllm_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1800,
        }
        headers = {"Authorization": f"Bearer {self.settings.vllm_api_key}"}
        url = self.settings.vllm_base_url.rstrip("/") + "/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            if not text:
                return LLMResult("", False, "vllm", "Модель вернула пустой ответ")
            return LLMResult(text=text, success=True, provider="vllm")
        except Exception as exc:
            logger.warning("llm_vllm_failed", extra={"event": "llm_vllm_failed", "error": str(exc), "success": False})
            return LLMResult("", False, "vllm", str(exc))


def build_llm_adapter(settings: Settings) -> LLMAdapter:
    if settings.llm_provider == "stub":
        logger.info("llm_adapter_selected", extra={"event": "llm_adapter_selected", "provider": "stub"})
        return StubLLMAdapter()
    logger.info("llm_adapter_selected", extra={"event": "llm_adapter_selected", "provider": "vllm"})
    return VLLMLLMAdapter(settings)
