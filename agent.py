

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from PIL import Image as PILImage

from ocr_pipeline import SuryaEngine
from field_merger import merge, FieldList, Field
from validation_engine import validate_field, validate_cross_field, SENSITIVE_FIELDS
from rag_setup import query_rag_field, query_rag

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL       = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 600
YOLO_WEIGHTS    = "D:/FormSaathi/runs/detect/formfields4/weights/best.pt"

# Max OCR lines to include in system prompt — enough context, not token-wasteful
MAX_OCR_LINES_IN_PROMPT = 60

# Fields that need sensitive handling — never send raw value to LLM
SENSITIVE_FIELD_KEYWORDS = {"aadhaar", "account", "income", "salary"}


# ── Session state ─────────────────────────────────────────────────────────────

@dataclass
class AgentSession:
    """
    Holds all state for one form-filling session.
    Persists across the full conversation.

    raw_ocr_context : actual text lines extracted from the uploaded form.
                      This is what the LLM uses to reason about the form.
                      form_type is a display label only — never used for inference.
    """
    field_list      : FieldList
    form_type       : str              # display label only — e.g. "PAN Form 49A"
    mode            : str              # "guided" or "free"
    history         : list[dict]       # OpenAI message history
    raw_ocr_context : str = ""         # ← actual OCR text from the uploaded form
    current_idx     : int = 0          # guided mode: which field we're on
    filled_fields   : dict = None

    def __post_init__(self):
        if self.filled_fields is None:
            self.filled_fields = {}

    @property
    def current_field(self) -> Optional[Field]:
        fields = self.field_list.fields
        if self.current_idx < len(fields):
            return fields[self.current_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_idx >= len(self.field_list.fields)


# ── OCR context extractor ─────────────────────────────────────────────────────

def extract_ocr_context(ocr_result, max_lines: int = MAX_OCR_LINES_IN_PROMPT) -> str:
    """
    Pull raw text lines from Surya OCR result and format them as a
    clean numbered list for the system prompt.

    This is the ONLY source of form knowledge the LLM is allowed to use.
    It does not interpret — it just transcribes what is literally on the form.
    """
    lines = []
    for i, line in enumerate(ocr_result.text_lines[:max_lines]):
        text = line.text.strip()
        if text:
            lines.append(f"  {i + 1:02d}. {text}")

    if not lines:
        return "  (No text could be extracted from this form)"

    return "\n".join(lines)


# ── System prompt builder ─────────────────────────────────────────────────────

def _build_system_prompt(
    field_list      : FieldList,
    form_type       : str,
    mode            : str,
    raw_ocr_context : str = "",
) -> str:
    """
    Build the system prompt.

    KEY DESIGN DECISION:
    The LLM is grounded ONLY in:
      1. The raw OCR text extracted from the uploaded form (raw_ocr_context)
      2. The detected field labels from YOLO + OCR (field_lines)

    The form_type label is shown for display context but the LLM is
    explicitly instructed NOT to use it to infer what fields mean.
    This prevents hallucination from wrong classification
    (e.g. "SBI Deposit Slip" misclassified as "SBI Account Opening Form").
    """

    # Detected fields summary — label + class
    field_lines = []
    for f in field_list.fields:
        label = f.label_text.strip() or f.ocr_text.strip() or f"Field {f.field_id}"
        field_lines.append(
            f"  [{f.field_id}] {f.class_name} | label: {label[:80]}"
        )
    fields_summary = "\n".join(field_lines) if field_lines else "  (no fields detected)"

    # OCR context block — this is the LLM's only factual reference
    ocr_block = raw_ocr_context if raw_ocr_context.strip() else "  (OCR text unavailable)"

    mode_instructions = {
        "guided": """
You are in GUIDED mode.
- Walk the user through the form field by field, one at a time.
- For each field: explain what it is, what to write, and why it matters.
- After explaining, ask the user to provide their value for that field.
- When the user provides a value, confirm it and move to the next field.
- If validation fails, explain the error clearly and ask them to re-enter.
- Track progress: tell the user which field number they are on (e.g. "Field 3 of 12").
""",
        "free": """
You are in FREE QUERY mode.
- The user has already uploaded the form. They can ask any question at any time.
- Answer questions about any field, what documents are needed, how to fill sections, etc.
- If the user provides a value to check (e.g. "is my PAN ABCPE1234F correct?"),
  validate it and give clear feedback.
- If the user asks about a specific field, explain it thoroughly.
- Do not force any order — answer whatever the user asks.
"""
    }

    return f"""You are FormSaathi, an AI assistant that helps Indian users fill government and bank forms.

DETECTED FORM LABEL: {form_type}
INTERACTION MODE: {mode.upper()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT — HOW TO USE THIS FORM LABEL:
The label above ("{form_type}") is auto-detected and MAY BE WRONG.
You must NEVER use this label to guess, infer, or assume what any field means.
Your ONLY source of truth about this form is the RAW OCR TEXT below.
If a field's purpose is not clear from the OCR text, say so honestly.
DO NOT invent context like "to open your account" or "for loan processing"
unless those exact words appear in the OCR text of this specific form.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{mode_instructions.get(mode, mode_instructions["free"])}

────────────────────────────────────────────────────
RAW TEXT EXTRACTED FROM THIS FORM (your ground truth):
────────────────────────────────────────────────────
{ocr_block}

────────────────────────────────────────────────────
FIELDS DETECTED ON THIS FORM (YOLO + OCR):
────────────────────────────────────────────────────
{fields_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES — never break these:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Use simple, plain language. No complex legal or bureaucratic jargon.

2. Use common abbreviations as-is: PAN, KYC, OTP, TDS, IFSC, DOB, AO.
   Only add the full form if it genuinely helps the user understand the field.

3. Give complete, thorough explanations. Never skip details. Never over-summarize.
   Cover every field and sub-field fully.

4. ONLY describe what each field literally asks for, based on its label text
   as it appears in the OCR text above.
   DO NOT invent purpose or context ("to open your account", "for loan processing",
   "for KYC verification") unless those exact words appear in the OCR text above.

5. If the form title or purpose is clearly visible in the OCR text, use that.
   If it is not clearly visible, tell the user:
   "I can see the fields on this form but the form title wasn't fully captured.
    Please check the top of your physical form for its name."

6. If you don't know something specific to this form, say so honestly and
   suggest the user check the official instructions booklet that came with the form.

7. Keep responses focused and concise — avoid unnecessary filler text.

8. Always be encouraging and patient. Many users are filling these forms
   for the first time.
"""


# ── RAG context fetcher ───────────────────────────────────────────────────────

def _get_rag_context(field: Field, form_type: str) -> str:
    """
    Fetch relevant context from RAG for a specific field.
    Returns empty string if RAG unavailable or irrelevant.
    RAG is used as supplementary info only — OCR text takes priority.
    """
    label = field.label_text.strip() or field.ocr_text.strip()
    if not label or len(label) < 3:
        return ""

    context = query_rag_field(label, form_type)
    if context and len(context) > 20:
        return f"\n[Supplementary info from knowledge base: {context}]"
    return ""


# ── Sensitive field check ─────────────────────────────────────────────────────

def _is_sensitive(field_label: str) -> bool:
    label_lower = field_label.lower()
    return any(kw in label_lower for kw in SENSITIVE_FIELD_KEYWORDS)


def _handle_sensitive_validation(field_label: str, value: str) -> str:
    """Validate sensitive field on-device, return only pass/fail message."""
    key = field_label.lower().replace(" ", "_")
    result = validate_field(key, value)
    if result.valid:
        return f"✓ Your {field_label} format is valid."
    else:
        return f"✗ {result.error} — {result.suggestion}"


# ── User message processor ────────────────────────────────────────────────────

def _extract_field_value(user_message: str, field: Field) -> Optional[str]:
    """
    Try to extract a field value from the user's message.
    Simple heuristic — if message is short and looks like a value, treat it as one.
    """
    msg = user_message.strip()
    if len(msg) < 60 and not msg.endswith("?"):
        return msg
    return None


def _build_user_turn(
    user_message    : str,
    session         : AgentSession,
    explicit_field  : Optional[Field] = None,
) -> str:
    """
    Augment the user's message with validation results and RAG context
    before sending to the LLM. The LLM never sees raw sensitive values.
    """
    augmented = user_message
    notes     = []

    field = explicit_field or session.current_field

    if field:
        label = field.label_text.strip() or field.ocr_text.strip() or f"Field {field.field_id}"

        value = _extract_field_value(user_message, field)

        if value and len(value) > 1:
            if _is_sensitive(label):
                validation_note = _handle_sensitive_validation(label, value)
                notes.append(f"[Device validation (sensitive field): {validation_note}]")
                notes.append(f"[Note: Raw value was NOT sent to AI — validated privately]")
                augmented = f"User provided value for '{label}' (value hidden for privacy)"

            else:
                result = validate_field(label.lower().replace(" ", "_"), value)
                if result.valid:
                    notes.append(f"[Validation: ✓ Valid — {result.suggestion or 'format correct'}]")
                    session.filled_fields[label] = value
                else:
                    notes.append(f"[Validation: ✗ {result.error}. Suggestion: {result.suggestion}]")

        # RAG is supplementary — labelled clearly so LLM knows it's not from the form
        rag_ctx = _get_rag_context(field, session.form_type)
        if rag_ctx:
            notes.append(rag_ctx)

    if notes:
        augmented = augmented + "\n\n" + "\n".join(notes)

    return augmented


# ── Guided mode helpers ───────────────────────────────────────────────────────

def _guided_field_prompt(field: Field, form_type: str, index: int, total: int) -> str:
    """Build the prompt that asks the agent to explain a specific field."""
    label   = field.label_text.strip() or field.ocr_text.strip() or f"Field {field.field_id}"
    rag_ctx = _get_rag_context(field, form_type)

    return (
        f"[GUIDED MODE — Field {index + 1} of {total}]\n"
        f"Field type : {field.class_name}\n"
        f"Field label: {label}\n"
        f"{rag_ctx}\n"
        f"Explain this field to the user based ONLY on what is visible in the form's "
        f"OCR text. Do not assume what this form is for. Ask the user to provide their value."
    )


def _guided_next(session: AgentSession, client: OpenAI) -> str:
    """Advance to next field in guided mode and get agent explanation."""
    if session.is_complete:
        return _run_completion_check(session, client)

    field   = session.current_field
    total   = len(session.field_list.fields)
    prompt  = _guided_field_prompt(field, session.form_type, session.current_idx, total)

    session.history.append({"role": "user", "content": prompt})
    response = _call_llm(session.history, client)
    session.history.append({"role": "assistant", "content": response})
    return response


def _run_completion_check(session: AgentSession, client: OpenAI) -> str:
    """Run cross-field validation at end of guided mode."""
    if not session.filled_fields:
        return "All fields completed! Please review your form before submitting."

    cross  = validate_cross_field(session.filled_fields, session.form_type)
    issues = cross.get("issues", [])

    if not issues:
        return (
            "✓ All fields completed and validated!\n\n"
            "No issues found. Your form looks good — please review once before submitting."
        )

    issue_lines = []
    for issue in issues:
        fields_str = " + ".join(issue.get("fields", []))
        issue_lines.append(f"• {fields_str}: {issue.get('suggestion', issue.get('issue', ''))}")

    return (
        "All fields completed. Found a few things to double-check:\n\n"
        + "\n".join(issue_lines)
        + "\n\nPlease correct these before submitting."
    )


# ── LLM caller ────────────────────────────────────────────────────────────────

def _call_llm(history: list[dict], client: OpenAI) -> str:
    """Send message history to GPT-4o-mini and return response text."""
    try:
        response = client.chat.completions.create(
            model       = LLM_MODEL,
            temperature = LLM_TEMPERATURE,
            max_tokens  = LLM_MAX_TOKENS,
            messages    = history,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Agent error: {e}]"


# ── Session initialiser ───────────────────────────────────────────────────────

def create_session(
    image_path   : str,
    mode         : str = "free",
    form_type    : str = "",
) -> AgentSession:
    """
    Full pipeline: load image → OCR → YOLO → merge → create session.

    Parameters
    ----------
    image_path : path to form page image (PNG/JPG)
    mode       : "guided" or "free"
    form_type  : optional display label hint (auto-detected if empty)
                 NOTE: This is a display label only. The LLM reasons from OCR text.
    """
    print(f"\n[agent]  Loading form: {image_path}")

    # OCR
    print("[agent]  Running OCR...")
    engine     = SuryaEngine()
    pil_img    = PILImage.open(image_path).convert("RGB")
    ocr_result = engine.ocr_batch([pil_img])[0]

    # Extract raw OCR context — this is the LLM's ground truth
    raw_ocr_context = extract_ocr_context(ocr_result)

    # Field detection + merge
    print("[agent]  Detecting and merging fields...")
    field_list = merge(image_path, ocr_result, YOLO_WEIGHTS)

    # Auto-detect display label from OCR text (used for display only)
    if not form_type:
        form_type = _detect_form_type_from_ocr(ocr_result)

    print(f"[agent]  Form label (display only): {form_type}")
    print(f"[agent]  Mode: {mode.upper()} | Fields: {len(field_list.fields)}")

    # Build system prompt — raw_ocr_context is the key addition
    system_prompt = _build_system_prompt(field_list, form_type, mode, raw_ocr_context)
    history = [{"role": "system", "content": system_prompt}]

    return AgentSession(
        field_list      = field_list,
        form_type       = form_type,
        mode            = mode,
        history         = history,
        raw_ocr_context = raw_ocr_context,
    )


def _detect_form_type_from_ocr(ocr_result) -> str:
    """
    Best-effort display label from OCR text.

    This is used ONLY for display in the UI header and in the system prompt
    as a labelled hint. The LLM is explicitly told NOT to reason from this label.

    Checks high-specificity keywords BEFORE generic institution names to avoid
    misclassifying sub-forms (e.g. SBI Deposit Slip → SBI Account Opening Form).
    """
    all_text = " ".join(
        line.text for line in ocr_result.text_lines[:20]
    ).lower()

    # ── Specific transaction slips — must come before generic bank check ──────
    if "pay in slip" in all_text or "pay-in slip" in all_text:
        return "Bank Pay-In Slip / Deposit Slip"
    if "deposit slip" in all_text or "deposit challan" in all_text:
        return "Bank Deposit Slip"
    if "withdrawal slip" in all_text or "withdrawal form" in all_text:
        return "Bank Withdrawal Slip"
    if "cheque" in all_text and ("requisition" in all_text or "request" in all_text):
        return "Cheque Book Requisition Form"
    if "rtgs" in all_text or "neft" in all_text:
        return "RTGS/NEFT Requisition Form"
    if "demand draft" in all_text or " dd " in all_text:
        return "Demand Draft Request Form"

    # ── Government / tax forms ────────────────────────────────────────────────
    if "49a" in all_text or "permanent account" in all_text:
        return "PAN Form 49A"
    if "aadhaar" in all_text or "uidai" in all_text:
        return "Aadhaar Enrollment Form"
    if "15g" in all_text:
        return "Form 15G"
    if "15h" in all_text:
        return "Form 15H"
    if "pm kisan" in all_text or "pm-kisan" in all_text:
        return "PM Kisan Form"
    if "ration" in all_text:
        return "Ration Card Form"
    if "passport" in all_text:
        return "Passport Application Form"
    if "income tax" in all_text:
        return "Income Tax Form"

    # ── Loan / KYC / generic bank ─────────────────────────────────────────────
    if "loan application" in all_text:
        return "Loan Application Form"
    if "kyc" in all_text:
        return "KYC Form"
    if "account opening" in all_text or "account open" in all_text:
        return "Bank Account Opening Form"

    # ── Generic institution fallback — checked last ───────────────────────────
    if "sbi" in all_text or "state bank" in all_text:
        return "SBI Form"
    if "hdfc" in all_text:
        return "HDFC Bank Form"
    if "icici" in all_text:
        return "ICICI Bank Form"
    if "pnb" in all_text or "punjab national" in all_text:
        return "PNB Form"
    if "post office" in all_text or "india post" in all_text:
        return "India Post Form"

    return "Government / Bank Form"


# ── Main chat function ────────────────────────────────────────────────────────

def chat(session: AgentSession, user_message: str, client: OpenAI) -> str:
    """
    Process one user message and return agent response.
    Handles both guided and free query modes.
    """
    user_message = user_message.strip()

    # ── Guided mode ───────────────────────────────────────────────────────────
    if session.mode == "guided":

        if user_message.lower() in ("next", "skip", "n"):
            session.current_idx += 1
            return _guided_next(session, client)

        if user_message.lower() in ("back", "previous", "b"):
            session.current_idx = max(0, session.current_idx - 1)
            return _guided_next(session, client)

        if user_message.lower() in ("restart", "reset"):
            session.current_idx = 0
            session.filled_fields = {}
            return _guided_next(session, client)

        field = session.current_field
        if field:
            augmented = _build_user_turn(user_message, session)
            session.history.append({"role": "user", "content": augmented})
            response = _call_llm(session.history, client)
            session.history.append({"role": "assistant", "content": response})

            if not user_message.endswith("?") and len(user_message) < 100:
                session.current_idx += 1
                if not session.is_complete:
                    next_response = _guided_next(session, client)
                    return response + "\n\n---\n\n" + next_response
                else:
                    completion = _run_completion_check(session, client)
                    return response + "\n\n---\n\n" + completion

            return response

        return _run_completion_check(session, client)

    # ── Free query mode ───────────────────────────────────────────────────────
    else:
        target_field = None
        msg_lower    = user_message.lower()

        for f in session.field_list.fields:
            label = (f.label_text + " " + f.ocr_text).lower()
            if label.strip() and any(
                word in msg_lower for word in label.split()
                if len(word) > 3
            ):
                target_field = f
                break

        augmented = _build_user_turn(user_message, session, explicit_field=target_field)
        session.history.append({"role": "user", "content": augmented})
        response = _call_llm(session.history, client)
        session.history.append({"role": "assistant", "content": response})
        return response


# ── CLI interface ─────────────────────────────────────────────────────────────

def run_cli(image_path: str, mode: str = "free", form_type: str = ""):
    """Interactive terminal chat for testing the agent."""

    if not OPENAI_API_KEY:
        print("[agent]  OPENAI_API_KEY not set. Add to .env file.")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)
    session = create_session(image_path, mode, form_type)

    print("\n" + "=" * 60)
    print(f"FormSaathi — {session.form_type}")
    print(f"Mode: {mode.upper()} | {len(session.field_list.fields)} fields detected")
    print("=" * 60)

    if mode == "guided":
        print("\nCommands: 'next'/'skip' — skip field | 'back' — previous field")
        print("          'restart' — start over | 'quit' — exit\n")
        first_response = _guided_next(session, client)
        print(f"\nFormSaathi: {first_response}\n")

    else:
        print("\nAsk anything about the form. Type 'quit' to exit.\n")
        greeting_prompt = (
            f"The user has uploaded a form. Based on the OCR text in your context, "
            f"identify what this form actually is from its content. "
            f"Greet them briefly, state what the form appears to be "
            f"(use the OCR text — do not rely on the form label), "
            f"mention how many fields were detected, and ask what they need help with."
        )
        session.history.append({"role": "user", "content": greeting_prompt})
        greeting = _call_llm(session.history, client)
        session.history.append({"role": "assistant", "content": greeting})
        print(f"\nFormSaathi: {greeting}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[agent]  Session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n[agent]  Session ended. Goodbye!")
            break

        response = chat(session, user_input, client)
        print(f"\nFormSaathi: {response}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FormSaathi Agent")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to form image (PNG/JPG)")
    parser.add_argument("--mode", type=str, default="free",
                        choices=["guided", "free"],
                        help="guided = field by field | free = ask anything (default: free)")
    parser.add_argument("--form", type=str, default="",
                        help="Form type display label hint (auto-detected if not set)")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"[agent]  Image not found: {args.image}")
        sys.exit(1)

    run_cli(
        image_path = args.image,
        mode       = args.mode,
        form_type  = args.form,
    )