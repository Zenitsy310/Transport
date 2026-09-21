from pydantic import BaseModel, Field


class ProcessDocumentResponse(BaseModel):
    request_id: str
    source_text: str
    answer_text: str
    ocr_method: str
    rag_status: str
    llm_provider: str
    warnings: list[str] = Field(default_factory=list)
    rag_sources: list[str] = Field(default_factory=list)


class ExportDocxRequest(BaseModel):
    answer_text: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    ocr_provider: str
    llm_provider: str
    rag_provider: str
