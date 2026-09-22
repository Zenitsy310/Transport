const fileInput = document.getElementById("file-input");
const processBtn = document.getElementById("process-btn");
const exportBtn = document.getElementById("export-btn");
const fileName = document.getElementById("file-name");
const statusEl = document.getElementById("status");
const sourceText = document.getElementById("source-text");
const answerText = document.getElementById("answer-text");
const preview = document.getElementById("preview");
const ocrMeta = document.getElementById("ocr-meta");
const genMeta = document.getElementById("gen-meta");
const healthEl = document.getElementById("health");

let selectedFile = null;
let requestId = null;
let previewUrl = null;

function setStatus(text, kind = "") {
  statusEl.hidden = !text;
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`.trim();
}

function formatDetail(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || item.detail || JSON.stringify(item)).join("; ");
  }
  return String(detail);
}

async function readApiPayload(response) {
  const raw = await response.text();
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      return JSON.parse(raw);
    } catch {
      throw new Error("Сервер вернул повреждённый JSON.");
    }
  }
  if (response.status === 413) {
    throw new Error("Файл слишком большой для загрузки (лимит 20 МБ).");
  }
  if (response.status === 502 || response.status === 503) {
    throw new Error("Сервис обработки недоступен. Проверьте, что backend запущен через docker compose.");
  }
  if (response.status === 504) {
    throw new Error("Превышено время ожидания обработки документа. Попробуйте файл меньшего размера.");
  }
  if (!raw.trim() || raw.trim().startsWith("<")) {
    throw new Error(`Сервер вернул HTML вместо данных (код ${response.status}). Чаще всего это падение backend или ошибка nginx.`);
  }
  throw new Error(raw.slice(0, 300) || `Ошибка ${response.status}`);
}

fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files[0] || null;
  processBtn.disabled = !selectedFile;
  exportBtn.disabled = true;
  requestId = null;
  answerText.value = "";
  sourceText.value = "";
  ocrMeta.textContent = "";
  genMeta.textContent = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  if (!selectedFile) {
    fileName.textContent = "Файл не выбран";
    preview.classList.add("empty");
    preview.textContent = "Предпросмотр появится после выбора файла";
    return;
  }
  fileName.textContent = selectedFile.name;
  previewUrl = URL.createObjectURL(selectedFile);
  preview.classList.remove("empty");
  preview.innerHTML = "";
  if (selectedFile.type.startsWith("image/")) {
    const img = document.createElement("img");
    img.src = previewUrl;
    img.alt = "Исходное изображение";
    preview.appendChild(img);
  } else if (selectedFile.type === "application/pdf") {
    const embed = document.createElement("embed");
    embed.src = previewUrl;
    embed.type = "application/pdf";
    preview.appendChild(embed);
  } else {
    preview.classList.add("empty");
    preview.textContent = "Предпросмотр недоступен для этого формата. После обработки текст появится ниже.";
  }
});

processBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  processBtn.disabled = true;
  setStatus("Обработка: OCR → поиск по НПА → генерация ответа…");
  const form = new FormData();
  form.append("file", selectedFile, selectedFile.name);
  try {
    const response = await fetch("/api/v1/process-document", { method: "POST", body: form });
    const payload = await readApiPayload(response);
    if (!response.ok) {
      throw new Error(formatDetail(payload.detail) || "Ошибка обработки");
    }
    requestId = payload.request_id;
    sourceText.value = payload.source_text;
    answerText.value = payload.answer_text;
    ocrMeta.textContent = `OCR: ${payload.ocr_method}`;
    genMeta.textContent = `LLM: ${payload.llm_provider} · RAG: ${payload.rag_status}`;
    exportBtn.disabled = !payload.answer_text;
    const warn = (payload.warnings || []).join(" ");
    setStatus(warn || "Черновик ответа готов. Проверьте текст перед экспортом.", warn ? "" : "ok");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    processBtn.disabled = !selectedFile;
  }
});

exportBtn.addEventListener("click", async () => {
  try {
    const response = await fetch("/api/v1/export-docx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer_text: answerText.value, request_id: requestId }),
    });
    if (!response.ok) {
      const payload = await readApiPayload(response);
      throw new Error(formatDetail(payload.detail) || "Не удалось сформировать DOCX");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "mosgortrans-reply.docx";
    link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    setStatus(error.message, "error");
  }
});

async function loadHealth() {
  try {
    const response = await fetch("/api/v1/health");
    const payload = await response.json();
    healthEl.textContent = `OCR: ${payload.ocr_provider} · LLM: ${payload.llm_provider} · RAG: ${payload.rag_provider}`;
  } catch {
    healthEl.textContent = "Сервис недоступен";
  }
}

loadHealth();
