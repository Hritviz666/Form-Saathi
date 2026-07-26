

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")

INSTRUCTIONS_DIR    = Path("Datasets/Instructions")   # put instruction PDFs here
CHROMA_PERSIST_DIR  = Path("data/chroma_db")          # persistent vector store
CHROMA_COLLECTION   = "form_instructions"

EMBED_MODEL         = "text-embedding-3-small"        # $0.02/1M tokens — very cheap
LLM_MODEL           = "gpt-4o-mini"
LLM_TEMPERATURE     = 0.0
LLM_MAX_TOKENS      = 512

CHUNK_SIZE          = 512    # tokens per chunk
CHUNK_OVERLAP       = 64     # overlap between chunks
TOP_K               = 4      # number of chunks to retrieve per query

# ── Lazy imports (heavy — only load when needed) ──────────────────────────────

def _get_llama_components():
    """Import LlamaIndex components. Deferred to avoid slow startup."""
    try:
        from llama_index.core import (
            VectorStoreIndex,
            SimpleDirectoryReader,
            StorageContext,
            Settings,
        )
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
        import chromadb

        return (VectorStoreIndex, SimpleDirectoryReader, StorageContext,
                Settings, SentenceSplitter, ChromaVectorStore,
                OpenAIEmbedding, LlamaOpenAI, chromadb)

    except ImportError as e:
        print(f"[rag]  Missing dependency: {e}")
        print("[rag]  Run: pip install llama-index llama-index-vector-stores-chroma "
              "llama-index-embeddings-openai llama-index-llms-openai chromadb")
        sys.exit(1)


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(force_rebuild: bool = False) -> None:
    """
    Load all PDFs from INSTRUCTIONS_DIR, chunk them, embed with
    text-embedding-3-small, and persist to ChromaDB.

    Parameters
    ----------
    force_rebuild : if True, drops existing collection and rebuilds from scratch
    """
    if not OPENAI_API_KEY:
        print("[rag]  OPENAI_API_KEY not set — cannot build index")
        sys.exit(1)

    (VectorStoreIndex, SimpleDirectoryReader, StorageContext,
     Settings, SentenceSplitter, ChromaVectorStore,
     OpenAIEmbedding, LlamaOpenAI, chromadb) = _get_llama_components()

    # Create instructions dir if missing
    INSTRUCTIONS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    # Check PDFs exist
    pdf_files = list(INSTRUCTIONS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[rag]  No PDFs found in {INSTRUCTIONS_DIR}/")
        print("[rag]  Add official form instruction PDFs to that folder and re-run.")
        print("[rag]  Example files:")
        print("         PAN_49A_instructions.pdf")
        print("         Aadhaar_enrollment_instructions.pdf")
        print("         SBI_account_opening_instructions.pdf")
        _create_fallback_index(
            VectorStoreIndex, StorageContext, Settings,
            ChromaVectorStore, OpenAIEmbedding, LlamaOpenAI, chromadb
        )
        return

    print(f"[rag]  Found {len(pdf_files)} instruction PDF(s):")
    for p in pdf_files:
        print(f"         {p.name}")

    # Configure LlamaIndex settings
    Settings.embed_model = OpenAIEmbedding(
        model   = EMBED_MODEL,
        api_key = OPENAI_API_KEY,
    )
    Settings.llm = LlamaOpenAI(
        model       = LLM_MODEL,
        temperature = LLM_TEMPERATURE,
        max_tokens  = LLM_MAX_TOKENS,
        api_key     = OPENAI_API_KEY,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
    )

    # ChromaDB setup
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    if force_rebuild:
        try:
            chroma_client.delete_collection(CHROMA_COLLECTION)
            print(f"[rag]  Dropped existing collection '{CHROMA_COLLECTION}'")
        except Exception:
            pass

    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)

    # Check if already indexed (skip if collection has documents)
    existing_count = chroma_collection.count()
    if existing_count > 0 and not force_rebuild:
        print(f"[rag]  Collection already has {existing_count} chunks — skipping rebuild.")
        print("[rag]  Use --build --force to rebuild from scratch.")
        return

    # Load PDFs
    print(f"\n[rag]  Loading PDFs from {INSTRUCTIONS_DIR}/...")
    documents = SimpleDirectoryReader(
        input_dir        = str(INSTRUCTIONS_DIR),
        required_exts    = [".pdf"],
        recursive        = False,
    ).load_data()

    print(f"[rag]  Loaded {len(documents)} document pages")

    # Build index (embeds + stores in ChromaDB)
    print(f"[rag]  Embedding chunks (model: {EMBED_MODEL})...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context = storage_context,
        show_progress   = True,
    )

    final_count = chroma_collection.count()
    print(f"\n[rag]  Index built — {final_count} chunks stored in {CHROMA_PERSIST_DIR}/")


def _create_fallback_index(
    VectorStoreIndex, StorageContext, Settings,
    ChromaVectorStore, OpenAIEmbedding, LlamaOpenAI, chromadb
) -> None:
    """
    If no instruction PDFs exist, seed ChromaDB with hardcoded knowledge
    about common Indian forms so the agent isn't completely blind.
    This is a bootstrap fallback — replace with real PDFs when available.
    """
    from llama_index.core import Document

    print("\n[rag]  No PDFs found — seeding fallback knowledge base...")

    fallback_docs = [
        Document(text="""
PAN Form 49A — Application for Allotment of Permanent Account Number
Applicable for: Indian citizens, Indian companies, entities incorporated in India.

Required documents:
- Proof of Identity: Aadhaar card, Passport, Voter ID, Driving License
- Proof of Address: Aadhaar card, Passport, Utility bill (not older than 3 months)
- Proof of Date of Birth: Birth certificate, Matriculation certificate, Passport

AO Code: Assessing Officer code. Consists of Area Code, AO Type, Range Code, AO Number.
For salaried individuals in cities, use the AO code for your employer's TDS circle.
You can find your AO code on the Income Tax India website using your city and category.

Full Name: Must match exactly as it appears on your proof of identity.
Initials are not permitted — full expanded name required.

Title: Shri (for males), Smt. (for married females), Kumari (for unmarried females),
M/s (for companies and firms).

Gender: Applicable only for individual applicants.

Date of Birth: For individuals, enter actual date of birth.
For companies, enter date of incorporation.
For partnerships, enter date of partnership deed.

Father's Name: Mandatory for all individual applicants except where mother is
a single parent and PAN is applied by furnishing the name of mother only.

PAN Card Name: You can choose abbreviated name for printing on PAN card.
""", metadata={"source": "PAN_49A", "form": "Form 49A"}),

        Document(text="""
Aadhaar Enrollment Form — UIDAI
Aadhaar is a 12-digit unique identity number issued by UIDAI to residents of India.

Fields:
- Full Name: As per proof of identity document
- Date of Birth: DD/MM/YYYY format. Verified or Declared (if no document)
- Gender: Male / Female / Transgender
- Address: Complete residential address with pincode
- Mobile Number: Optional but recommended for OTP-based services
- Email: Optional

Proof of Identity accepted: Passport, PAN card, Voter ID, Driving License, 
Government photo ID cards.

Proof of Address accepted: Passport, Bank statement, Utility bill, 
Ration card, Voter ID with address.

Biometrics: All 10 fingerprints and iris scan captured at enrollment center.
Children below 5: Only photograph, no biometrics. Re-enrollment mandatory at age 5 and 15.
""", metadata={"source": "Aadhaar", "form": "Aadhaar Enrollment"}),

        Document(text="""
SBI Account Opening Form — State Bank of India
For opening Savings / Current accounts at SBI branches.

Fields:
- Name: Full name as per KYC documents
- Date of Birth: Mandatory for individual accounts
- PAN: Mandatory for accounts with transactions above Rs. 50,000
- Aadhaar: For KYC linkage (mandatory as per RBI guidelines)
- IFSC Code: 11-character code — first 4 letters are bank code, 5th is 0, last 6 are branch
- Nominee: Recommended — name, relationship, date of birth of nominee
- Mode of Operation: Single / Jointly / Either or Survivor / Anyone or Survivor

KYC documents required:
- Proof of Identity: PAN, Aadhaar, Passport, Voter ID
- Proof of Address: Aadhaar, Utility bill, Bank statement, Passport

Initial deposit: Minimum Rs. 500 for urban branches, Rs. 250 for rural branches (no-frills account: zero).
""", metadata={"source": "SBI_account", "form": "SBI Account Opening"}),

        Document(text="""
Form 15G — Declaration for Non-Deduction of TDS
Applicable for: Individuals below 60 years whose income is below taxable limit.
Form 15H is for senior citizens (60 years and above).

Purpose: Submitted to bank or financial institution to prevent TDS deduction on 
interest income when total income is below the basic exemption limit.

Fields:
- Name: As per PAN
- PAN: Mandatory — declaration is invalid without PAN
- Assessment Year: The year for which declaration is made (e.g. 2024-25)
- Residential Status: Resident Indian only — NRIs cannot submit Form 15G
- Nature of Income: Interest on deposits, NSC, etc.
- Estimated Income: Total estimated income for the year including the income for which declaration is made
- Previous Year Total Income: Should be below taxable limit (Rs. 2.5 lakh for individuals below 60)

Important: Filing false declaration is punishable under Section 277 of Income Tax Act.
""", metadata={"source": "Form15G", "form": "Form 15G"}),
    ]

    Settings.embed_model = OpenAIEmbedding(
        model   = EMBED_MODEL,
        api_key = OPENAI_API_KEY,
    )
    Settings.llm = LlamaOpenAI(
        model       = LLM_MODEL,
        temperature = LLM_TEMPERATURE,
        max_tokens  = LLM_MAX_TOKENS,
        api_key     = OPENAI_API_KEY,
    )

    chroma_client     = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        fallback_docs,
        storage_context = storage_context,
        show_progress   = True,
    )
    print(f"[rag]  Fallback index built — {chroma_collection.count()} chunks stored")
    print("[rag]  Add real PDFs to Datasets/Instructions/ and run --build --force to upgrade")


# ── Query engine ──────────────────────────────────────────────────────────────

def _load_query_engine():
    """Load the persisted index and return a query engine."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    (VectorStoreIndex, _, StorageContext,
     Settings, _, ChromaVectorStore,
     OpenAIEmbedding, LlamaOpenAI, chromadb) = _get_llama_components()

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    Settings.embed_model = OpenAIEmbedding(
        model   = EMBED_MODEL,
        api_key = OPENAI_API_KEY,
    )
    Settings.llm = LlamaOpenAI(
        model       = LLM_MODEL,
        temperature = LLM_TEMPERATURE,
        max_tokens  = LLM_MAX_TOKENS,
        api_key     = OPENAI_API_KEY,
    )

    chroma_client     = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

    if chroma_collection.count() == 0:
        raise RuntimeError(
            "RAG index is empty. Run: python rag_setup.py --build"
        )

    vector_store    = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index           = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )

    return index.as_query_engine(
        similarity_top_k = TOP_K,
        streaming        = False,
    )


def query_rag(question: str) -> str:
    """
    Query the RAG index with a question about form instructions.
    Returns answer string. Returns empty string if index unavailable.

    This is the function imported by agent.py.

    Usage:
        answer = query_rag("What documents are needed for PAN Form 49A?")
        answer = query_rag("What is AO code in PAN form?")
    """
    if not OPENAI_API_KEY:
        return ""

    try:
        engine   = _load_query_engine()
        response = engine.query(question)
        return str(response).strip()
    except RuntimeError as e:
        print(f"[rag]  {e}")
        return ""
    except Exception as e:
        print(f"[rag]  Query failed: {e}")
        return ""


def query_rag_field(field_label: str, form_type: str = "") -> str:
    """
    Specialised query for a specific field label.
    Used by agent.py when explaining a field to the user.

    Usage:
        context = query_rag_field("AO Code", form_type="PAN Form 49A")
        context = query_rag_field("Nominee", form_type="SBI Account Opening")
    """
    form_context = f" in {form_type}" if form_type else ""
    question     = (
        f"What is '{field_label}'{form_context}? "
        f"What should the applicant write here? "
        f"What documents or information are needed?"
    )
    return query_rag(question)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FormSaathi RAG Setup")
    parser.add_argument("--build", action="store_true",
                        help="Build or update the RAG index from PDFs")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if index already exists")
    parser.add_argument("--query", type=str, default="",
                        help="Query the RAG index")
    parser.add_argument("--test", action="store_true",
                        help="Run built-in test queries")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[rag]  OPENAI_API_KEY not set.")
        print("[rag]  Run: $env:OPENAI_API_KEY = 'sk-...'")
        sys.exit(1)

    if args.build:
        build_index(force_rebuild=args.force)

    if args.query:
        print(f"\n[query]  {args.query}")
        print("-" * 60)
        answer = query_rag(args.query)
        if answer:
            print(answer)
        else:
            print("No answer found. Run --build first.")

    if args.test:
        test_questions = [
            "What documents are needed for PAN Form 49A?",
            "What is AO code and how do I find it?",
            "What is the difference between Form 15G and Form 15H?",
            "What is IFSC code and where do I find it?",
            "Can a single parent apply for PAN without father's name?",
        ]
        print("\n── RAG Test Queries ──────────────────────────────────────")
        for q in test_questions:
            print(f"\nQ: {q}")
            print("-" * 50)
            answer = query_rag(q)
            print(answer if answer else "No answer — run --build first")

    if not any([args.build, args.query, args.test]):
        parser.print_help()