# Mosgortrans appeals MVP

Сквозной сценарий: загрузка файла → OCR/извлечение текста → гибридный RAG (Qdrant + LlamaIndex) → черновик ответа (Qwen2.5 через vLLM) → правка → экспорт DOCX.

## Состав

- `backend/` — FastAPI, адаптеры OCR / RAG / LLM, конвейер, экспорт DOCX
- `frontend/` — раздельный экран исходного документа и редактора ответа
- `data/knowledge_base/` — стартовые материалы для индексации

Авторизация, СЭД и аналитика в MVP не входят.

## Запуск

1. Скопируйте `.env.example` в `.env`.
2. Для демо без GPU можно выставить:
   - `OCR_PROVIDER=stub`
   - `LLM_PROVIDER=stub`
   - `RAG_PROVIDER=stub` (или оставить `llamaindex`, если доступен Qdrant)
3. Соберите стек: `docker compose up --build`
4. Откройте http://localhost:3000

PaddleOCR: `INSTALL_OCR=1` и `OCR_PROVIDER=paddle`.
vLLM (Qwen2.5-14B-Instruct) должен быть доступен по `VLLM_BASE_URL` (OpenAI-compatible `/v1`).

## API

- `GET /api/v1/health`
- `POST /api/v1/process-document` — multipart `file`
- `POST /api/v1/export-docx` — JSON `{ "answer_text", "request_id?" }`

Таймауты: OCR 30 с, LLM 45 с. Если генерация срывается, интерфейс получает распознанный текст и сообщение об ошибке.
