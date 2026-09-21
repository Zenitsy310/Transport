from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "mosgortrans-appeals"
    log_level: str = "INFO"

    ocr_provider: str = "paddle"  # paddle | stub
    llm_provider: str = "vllm"  # vllm | stub
    rag_provider: str = "llamaindex"  # llamaindex | stub

    ocr_timeout_seconds: float = 30.0
    llm_timeout_seconds: float = 45.0
    rag_timeout_seconds: float = 20.0

    max_upload_bytes: int = 20 * 1024 * 1024
    allowed_extensions: str = ".pdf,.jpg,.jpeg,.png,.txt,.docx"

    knowledge_base_dir: Path = Path("data/knowledge_base")
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "mosgortrans_npa"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rag_top_k: int = 5

    vllm_base_url: str = "http://host.docker.internal:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-14B-Instruct"
    vllm_api_key: str = "not-needed"

    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def allowed_suffixes(self) -> set[str]:
        return {item.strip().lower() for item in self.allowed_extensions.split(",") if item.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
