

from __future__ import annotations

import os
import uuid
import tempfile
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import cv2
from PIL import Image as PILImage

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from ocr_pipeline import SuryaEngine
from field_merger import merge, FieldList, Field, YOLO_WEIGHTS
from validation_engine import validate_field
from agent import (
    AgentSession,
    chat,
    _build_system_prompt,
    _guided_next,
    _call_llm,
    extract_ocr_context,           # ← new import
    _detect_form_type_from_ocr,    # ← replaces old inline logic
)
from rag_setup import query_rag

# ── Global state ──────────────────────────────────────────────────────────────

SESSIONS           : dict[str, AgentSession] = {}
SESSION_CREATED_AT : dict[str, str]          = {}
OPENAI_CLIENT      : Optional[OpenAI]        = None
SURYA_ENGINE       : Optional[SuryaEngine]   = None

TEMP_DIR    = Path(tempfile.gettempdir()) / "formsaathi_uploads"
PDF_DPI     = 300
MAX_FILE_MB = 20
MAX_PAGES   = 10


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global OPENAI_CLIENT, SURYA_ENGINE

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[startup]  WARNING: OPENAI_API_KEY not set — agent will not work")
    else:
        OPENAI_CLIENT = OpenAI(api_key=api_key)
        print("[startup]  OpenAI client ready")

    print("[startup]  Loading Surya OCR models (first load ~30–60s)...")
    SURYA_ENGINE = SuryaEngine()
    print("[startup]  Surya OCR ready")
    print("[startup]  FormSaathi API is up")

    yield

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("[shutdown]  Temp files cleaned up")


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "FormSaathi API",
    description = "Multimodal AI agent for Indian government and bank forms",
    version     = "2.1.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id  : str
    form_type   : str
    mode        : str
    page_count  : int
    field_count : int
    message     : str


class ChatRequest(BaseModel):
    session_id  : str
    message     : str
    mode        : Optional[str] = None


class ChatResponse(BaseModel):
    session_id    : str
    response      : str
    mode          : str
    current_field : Optional[int]
    total_fields  : int
    is_complete   : bool


class SessionState(BaseModel):
    session_id    : str
    form_type     : str
    mode          : str
    page_count    : int
    field_count   : int
    current_field : Optional[int]
    filled_fields : dict
    is_complete   : bool
    created_at    : str


class ValidateRequest(BaseModel):
    field_name : str
    value      : str


class ValidateResponse(BaseModel):
    field_name : str
    valid      : bool
    error      : Optional[str]
    suggestion : Optional[str]
    sensitive  : bool


class HealthResponse(BaseModel):
    status    : str
    ocr_ready : bool
    llm_ready : bool
    sessions  : int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_file_size(file: UploadFile, max_mb: int = MAX_FILE_MB):
    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)
    if size_mb > max_mb:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Maximum is {max_mb}MB.")


def _pdf_to_images(pdf_path: str, dpi: int = PDF_DPI) -> list[str]:
    """
    Convert PDF pages to PNG images using PyMuPDF.
    fitz imported lazily — avoids DLL crash at server startup on Windows.
    """
    try:
        import fitz
    except ImportError:
        raise HTTPException(
            500,
            "PDF support unavailable — PyMuPDF DLL error. Upload PNG or JPG instead."
        )

    doc        = fitz.open(pdf_path)
    page_count = min(len(doc), MAX_PAGES)
    mat        = fitz.Matrix(dpi / 72, dpi / 72)
    stem       = Path(pdf_path).stem
    out_dir    = TEMP_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for page_idx in range(page_count):
        page     = doc[page_idx]
        pix      = page.get_pixmap(matrix=mat, alpha=False)
        img_path = str(out_dir / f"page_{page_idx + 1:03d}.png")
        pix.save(img_path)
        image_paths.append(img_path)

    doc.close()
    print(f"[pdf]  Converted {page_count} pages from {Path(pdf_path).name}")
    return image_paths


def _merge_field_lists(field_lists: list[FieldList]) -> FieldList:
    """Combine FieldLists from multiple pages. field_id is globally unique."""
    if len(field_lists) == 1:
        return field_lists[0]

    all_fields = []
    global_id  = 0
    for fl in field_lists:
        for f in fl.fields:
            all_fields.append(Field(
                field_id   = global_id,
                class_name = f.class_name,
                class_id   = f.class_id,
                bbox       = f.bbox,
                conf       = f.conf,
                ocr_text   = f.ocr_text,
                label_text = f.label_text,
                page       = f.page,
            ))
            global_id += 1

    return FieldList(
        page       = 0,
        image_path = field_lists[0].image_path,
        fields     = all_fields,
        raw_ocr    = [line for fl in field_lists for line in fl.raw_ocr],
    )


def _process_upload(file_path: str) -> tuple[FieldList, list[str], list]:
    """
    Run full OCR + YOLO + merge pipeline. Handles images and PDFs.

    Returns
    -------
    field_list   : merged FieldList
    image_paths  : list of processed page image paths
    ocr_results  : list of raw OCR result objects (one per page)
                   — kept so we can pass full OCR text to the session
    """
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        image_paths = _pdf_to_images(file_path)
    elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
        image_paths = [file_path]
    else:
        raise HTTPException(415, f"Unsupported file type: {suffix}. Use PDF, PNG, or JPG.")

    field_lists  = []
    ocr_results  = []

    for page_idx, img_path in enumerate(image_paths):
        pil_img    = PILImage.open(img_path).convert("RGB")
        ocr_result = SURYA_ENGINE.ocr_batch([pil_img])[0]
        fl         = merge(img_path, ocr_result, YOLO_WEIGHTS, page=page_idx)
        field_lists.append(fl)
        ocr_results.append(ocr_result)

    return _merge_field_lists(field_lists), image_paths, ocr_results


def _build_combined_ocr_context(ocr_results: list) -> str:
    """
    Combine OCR text from all pages into one context string.
    Each page is labelled so the LLM can tell them apart on multi-page forms.
    """
    if len(ocr_results) == 1:
        return extract_ocr_context(ocr_results[0])

    parts = []
    for i, ocr_result in enumerate(ocr_results):
        page_text = extract_ocr_context(ocr_result)
        parts.append(f"--- Page {i + 1} ---\n{page_text}")

    return "\n\n".join(parts)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status    = "ok",
        ocr_ready = SURYA_ENGINE is not None,
        llm_ready = OPENAI_CLIENT is not None,
        sessions  = len(SESSIONS),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_form(
    file : UploadFile = File(...),
    mode : str        = Form(default="free"),
):
    """
    Upload a form image or PDF.
    Runs OCR + field detection + merge.
    Returns session_id and first agent message.

    The session is grounded in raw OCR text — not the form_type label.
    form_type is auto-detected for display purposes only.
    """
    if SURYA_ENGINE is None:
        raise HTTPException(503, "OCR engine not ready — server is still starting up")
    if OPENAI_CLIENT is None:
        raise HTTPException(503, "LLM not ready — OPENAI_API_KEY not set")
    if mode not in ("free", "guided"):
        raise HTTPException(400, "mode must be 'free' or 'guided'")

    _check_file_size(file)

    session_id = str(uuid.uuid4())
    suffix     = Path(file.filename or "form.png").suffix.lower() or ".png"
    save_path  = str(TEMP_DIR / f"{session_id}{suffix}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"[upload]  {file.filename} → session {session_id[:8]}...")

    try:
        field_list, image_paths, ocr_results = _process_upload(save_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")

    # form_type: display label only — detected from OCR keywords
    # It is shown in the UI and passed to the system prompt as a labelled hint,
    # but the LLM is explicitly instructed not to reason from it.
    form_type = _detect_form_type_from_ocr(ocr_results[0])

    # raw_ocr_context: the LLM's actual ground truth about what is on this form
    raw_ocr_context = _build_combined_ocr_context(ocr_results)

    system_prompt = _build_system_prompt(field_list, form_type, mode, raw_ocr_context)
    history       = [{"role": "system", "content": system_prompt}]

    session = AgentSession(
        field_list      = field_list,
        form_type       = form_type,
        mode            = mode,
        history         = history,
        raw_ocr_context = raw_ocr_context,
    )
    SESSIONS[session_id]           = session
    SESSION_CREATED_AT[session_id] = datetime.now().isoformat()

    # Generate first message
    if mode == "guided":
        first_message = _guided_next(session, OPENAI_CLIENT)
    else:
        # Ask the LLM to introduce the form based on what it actually reads
        # — not based on the form_type label
        greeting_prompt = (
            "The user has just uploaded a form. "
            "Based on the OCR text in your system context, identify what this form is "
            "from its actual content — the title, heading, or printed name on the form. "
            "Do not rely on the DETECTED FORM LABEL. "
            f"Greet them briefly, state what the form appears to be (from OCR text), "
            f"mention that {len(field_list.fields)} fields were detected, "
            "and ask what they need help with."
        )
        session.history.append({"role": "user", "content": greeting_prompt})
        first_message = _call_llm(session.history, OPENAI_CLIENT)
        session.history.append({"role": "assistant", "content": first_message})

    return UploadResponse(
        session_id  = session_id,
        form_type   = form_type,
        mode        = mode,
        page_count  = len(image_paths),
        field_count = len(field_list.fields),
        message     = first_message,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Send a message to the agent for an active session."""
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(404, f"Session '{req.session_id}' not found or expired")
    if OPENAI_CLIENT is None:
        raise HTTPException(503, "LLM not ready — OPENAI_API_KEY not set")
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    if req.mode and req.mode in ("free", "guided"):
        session.mode = req.mode

    response = chat(session, req.message, OPENAI_CLIENT)

    return ChatResponse(
        session_id    = req.session_id,
        response      = response,
        mode          = session.mode,
        current_field = session.current_idx if session.mode == "guided" else None,
        total_fields  = len(session.field_list.fields),
        is_complete   = session.is_complete,
    )


@app.get("/session/{session_id}", response_model=SessionState)
async def get_session(session_id: str):
    """Return current session state."""
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")

    return SessionState(
        session_id    = session_id,
        form_type     = session.form_type,
        mode          = session.mode,
        page_count    = len(set(f.page for f in session.field_list.fields)),
        field_count   = len(session.field_list.fields),
        current_field = session.current_idx if session.mode == "guided" else None,
        filled_fields = session.filled_fields,
        is_complete   = session.is_complete,
        created_at    = SESSION_CREATED_AT.get(session_id, ""),
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Remove session from memory and clean up temp files."""
    if session_id not in SESSIONS:
        raise HTTPException(404, f"Session '{session_id}' not found")

    del SESSIONS[session_id]
    SESSION_CREATED_AT.pop(session_id, None)

    for f in TEMP_DIR.glob(f"{session_id}*"):
        try:
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
        except Exception:
            pass

    return {"status": "deleted", "session_id": session_id}


@app.post("/validate", response_model=ValidateResponse)
async def validate_endpoint(req: ValidateRequest):
    """
    Validate a single field value independently.
    Frontend can call this for live validation as the user types.
    Sensitive fields validated on-device — raw value never stored.
    """
    result = validate_field(req.field_name, req.value)
    return ValidateResponse(
        field_name = req.field_name,
        valid      = result.valid,
        error      = result.error,
        suggestion = result.suggestion,
        sensitive  = result.sensitive,
    )


@app.get("/rag")
async def rag_query(q: str):
    """
    Query the RAG knowledge base directly.
    GET /rag?q=What is AO code in PAN form?
    """
    if not q.strip():
        raise HTTPException(400, "Query 'q' cannot be empty")

    answer = query_rag(q)
    return {
        "query"  : q,
        "answer" : answer or "No relevant information found.",
    }


@app.get("/sessions")
async def list_sessions():
    """List all active sessions (for debugging)."""
    return {
        "count"    : len(SESSIONS),
        "sessions" : [
            {
                "session_id" : sid,
                "form_type"  : s.form_type,
                "mode"       : s.mode,
                "fields"     : len(s.field_list.fields),
                "created_at" : SESSION_CREATED_AT.get(sid, ""),
            }
            for sid, s in SESSIONS.items()
        ],
    }