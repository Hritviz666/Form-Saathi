# FormSaathi 🇮🇳

> **Multimodal AI agent that helps Indian users fill government and bank forms — field by field, in plain language.**

FormSaathi combines a custom-trained object detection model, OCR, retrieval-augmented generation, and a validation engine into a single end-to-end pipeline. Upload any form (PAN, Aadhaar, EPF, bank slips, KYC) and the agent either walks you through it field by field or answers free-form questions about it.

---

## Demo

| Upload Screen | Guided Mode | Free Query Mode |
|---|---|---|
| Drag-drop PDF or image | Field-by-field explanation + typo correction | Ask anything about any field |

**Tested on:** PAN Form 49A · Aadhaar Enrollment · EPF Form 19 · SBI Pay-In Slip · Form 15G · PM Kisan · Ration Card · Passport Application

---

## Architecture

```
User uploads form (PDF / PNG / JPG)
        │
        ▼
┌─────────────────────────────────────────────────────┐ 
│                   FastAPI Backend                   │
│                                                     │
│  PDF → PyMuPDF → page images (300 DPI)              │
│                        │                            │
│              ┌─────────┴──────────┐                 │
│              ▼                    ▼                 │
│        Surya OCR            YOLOv8n                 │
│     (text extraction)   (field detection)           │
│              │                    │                 │
│              └─────────┬──────────┘                 │
│                        ▼                            │
│                  Field Merger                       │
│           (bbox-aware OCR ↔ YOLO merge)             │
│                        │                            │
│                        ▼                            │
│              Validation Engine                      │
│        (19 rules: PAN, IFSC, Aadhaar,               │
│         DOB, pincode, mobile, email…)               │
│                        │                            │
│                        ▼                            │
│         RAG  (LlamaIndex + ChromaDB)                │
│    (official form instructions knowledge base)      │
│                        │                            │
│                        ▼                            │
│           GPT-4o-mini Agent                         │
│   grounded in raw OCR text — not form label         │
│   (guided field-by-field OR free query mode)        │
└─────────────────────────────────────────────────────┘
        │
        ▼
React PWA Frontend (Vite + Tailwind CSS)
  UploadScreen → ChatScreen → dual-mode chat UI
```

---

## Key Design Decisions

**OCR-grounded LLM (anti-hallucination)**
The LLM receives the raw OCR text extracted from the uploaded form as its ground truth. The auto-detected form label is shown for display only and the model is explicitly instructed not to reason from it. This prevents hallucination on misclassified forms (e.g. SBI Deposit Slip being treated as an Account Opening Form).

**On-device sensitive field validation**
Fields containing Aadhaar numbers, account numbers, income, and salary are validated locally. Raw values are never sent to the LLM — only a pass/fail result is forwarded.

**Graceful degradation on unseen forms**
YOLOv8n was trained on a specific set of form types. On out-of-distribution forms, the model flags low-confidence detections and asks the user to confirm rather than silently passing wrong data downstream.

---

## ML Pipeline

| Component | Detail |
|---|---|
| **Object Detection** | YOLOv8n fine-tuned on FUNSD + 9 Indian government forms |
| **Training resolution** | 640×640, AMP enabled |
| **mAP@0.5** | 0.806 |
| **OCR** | Surya OCR with adaptive tiling for high-res forms |
| **PDF conversion** | PyMuPDF at 300 DPI (matches training resolution) |
| **Field classes** | text_field, checkbox, signature, date_field, dropdown, label |
| **Augmentation** | bbox-aware: rotation, brightness, JPEG compression, partial occlusion |

---

## Tech Stack

**Backend**
- Python 3.11
- FastAPI + Uvicorn
- YOLOv8n (Ultralytics)
- Surya OCR
- GPT-4o-mini (OpenAI)
- LlamaIndex + ChromaDB (RAG)
- PyMuPDF (PDF processing)

**Frontend**
- React 18 + Vite
- Tailwind CSS
- PWA-ready

**Infrastructure**
- CUDA (torch 2.3.0+cu121)
- REST API — session-based, stateless storage

---

## Project Structure

```
D:\FormSaathi\
│
├── agent.py              # GPT-4o-mini agent — guided + free query modes
├── main.py               # FastAPI backend — all REST endpoints
├── ocr_pipeline.py       # Surya OCR engine wrapper
├── field_merger.py       # YOLO + OCR bbox merge logic
├── validation_engine.py  # 19-rule field validator
├── rag_setup.py          # LlamaIndex + ChromaDB RAG setup
│
├── data/
│   ├── forms/            # Training form images
│   └── annotations/      # YOLO format labels
│
├── runs/
│   └── detect/
│       └── formfields4/
│           └── weights/
│               └── best.pt   # Fine-tuned YOLOv8n weights
│
├── formsaathi-frontend/
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── components/
│           ├── UploadScreen.jsx
│           ├── ChatScreen.jsx
│           ├── ChatBubble.jsx
│           ├── ValidationInput.jsx
│           ├── ProgressBar.jsx
│           ├── GuidedControls.jsx
│           ├── SessionHeader.jsx
│           └── TypingIndicator.jsx
│
└── .env                  # OPENAI_API_KEY
```

---

## Setup & Run

### Prerequisites
- Python 3.11
- Node.js 18+
- CUDA-capable GPU (recommended) or CPU
- OpenAI API key

### Backend

```bash
# Clone and enter project
cd D:\FormSaathi

# Create and activate virtual environment
python -m venv Saathi
Saathi\Scripts\activate

# Install dependencies
pip install fastapi uvicorn openai python-dotenv pillow opencv-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics surya-ocr pymupdf llama-index chromadb

# Add your OpenAI API key
echo OPENAI_API_KEY=your_key_here > .env

# Start the server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd D:\FormSaathi\formsaathi-frontend
npm install
npm run dev
```

Open `http://localhost:5173`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload form image/PDF, returns session_id + first agent message |
| `POST` | `/chat` | Send message to agent for active session |
| `GET` | `/session/{id}` | Get current session state |
| `DELETE` | `/session/{id}` | Clean up session |
| `POST` | `/validate` | Standalone field validator |
| `GET` | `/rag?q=...` | Query RAG knowledge base directly |
| `GET` | `/health` | Health check |

### Quick test

```bash
# Health check
curl http://localhost:8000/health

# Upload a form
curl -X POST http://localhost:8000/upload \
  -F "file=@form.pdf" \
  -F "mode=guided"

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_ID", "message": "What is AO code?"}'
```

---

## Validation Rules

The validation engine covers 19 field types with format checking and cross-field validation:

`PAN` · `Aadhaar` · `IFSC` · `mobile` · `email` · `pincode` · `date of birth` · `name` · `account number` · `amount` · `cheque number` · `income` · `TDS` · `nominee age` · `percentage` · `year` · `address` · `gender` · `signature`

Sensitive fields (Aadhaar, account number, income, salary) are validated on-device — raw values are never forwarded to the LLM.

---

## Limitations & Future Work

- **Field detection on unseen forms:** YOLOv8n was trained on 9 form types. Accuracy on out-of-distribution forms varies; expanding the training set would improve coverage.
- **Multi-language support:** Currently English only. Hindi and regional language OCR is a planned extension.
- **Session persistence:** Sessions are held in memory — server restart clears all sessions. A Redis or SQLite backend would fix this for production.
- **Voice input:** Text-only interface for now; a voice agent mode is planned.

---

## Author

**Shadow** (Hritviz Manral)  
B.Tech Information Technology · IIIT Una · 
[linkedin.com/in/hritvizmanral](https://linkedin.com/in/hritvizmanral)

---

## License

MIT