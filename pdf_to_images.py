

import os
import time
import logging
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG — edit these paths before running
# ─────────────────────────────────────────────

PDF_INPUT_DIR  = "Datasets"       # folder containing your 9 PDF files
IMAGE_OUTPUT_DIR = "data/forms/images"   # output folder (created if missing)

DPI = 300                                # 300 is standard for OCR training data
OUTPUT_FORMAT = "png"                    # png = lossless, no artifacts
SKIP_EXISTING = True                     # set False to force reconvert everything

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("pdf_to_images")


# ─────────────────────────────────────────────
# CORE FUNCTION
# ─────────────────────────────────────────────

def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = DPI,
    skip_existing: bool = SKIP_EXISTING,
) -> list[Path]:
    """
    Convert a single PDF to a list of PNG images (one per page).

    Args:
        pdf_path:      Path to the input PDF file.
        output_dir:    Directory where images will be saved.
                       A subfolder named after the PDF stem is created inside.
        dpi:           Resolution for rasterization (300 recommended for OCR).
        skip_existing: If True, skip pages whose output file already exists.

    Returns:
        List of output image Paths that were actually written.

    Raises:
        ValueError: If the PDF is encrypted and cannot be opened.
        RuntimeError: If PyMuPDF fails to render any page.
    """
    pdf_stem = pdf_path.stem                          # e.g. "Form49A"
    form_out_dir = output_dir / pdf_stem              # data/forms/images/Form49A/
    form_out_dir.mkdir(parents=True, exist_ok=True)

    # zoom factor: 72 DPI is PyMuPDF's internal base
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    written: list[Path] = []

    try:
        doc = fitz.open(str(pdf_path))
    except fitz.FileDataError as e:
        raise RuntimeError(f"Cannot open PDF '{pdf_path.name}': {e}") from e

    if doc.is_encrypted:
        raise ValueError(
            f"PDF '{pdf_path.name}' is encrypted. "
            "Decrypt it first (try: qpdf --decrypt input.pdf output.pdf)"
        )

    for page_num in range(len(doc)):
        out_filename = f"{pdf_stem}_page{page_num + 1:03d}.{OUTPUT_FORMAT}"
        out_path = form_out_dir / out_filename

        if skip_existing and out_path.exists():
            log.debug("  Skipping existing: %s", out_path.name)
            continue

        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix, alpha=False)  # alpha=False → RGB

        pix.save(str(out_path))
        written.append(out_path)
        log.debug("  Wrote: %s (%dx%d px)", out_path.name, pix.width, pix.height)

    doc.close()
    return written


def convert_all_pdfs(
    input_dir: str | Path = PDF_INPUT_DIR,
    output_dir: str | Path = IMAGE_OUTPUT_DIR,
    dpi: int = DPI,
    skip_existing: bool = SKIP_EXISTING,
) -> dict:
    """
    Convert all PDFs in input_dir to images, placing results in output_dir.

    Returns a summary dict:
        {
            "total_pdfs": int,
            "total_pages": int,
            "skipped_pdfs": list[str],   # encrypted or unreadable
            "output_dir": str,
            "elapsed_sec": float,
        }
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDF files found in: %s", input_dir)
        return {}

    log.info("Found %d PDF(s) in '%s'", len(pdf_files), input_dir)
    log.info("Output dir: %s  |  DPI: %d  |  Format: %s", output_dir, dpi, OUTPUT_FORMAT.upper())

    total_pages = 0
    skipped_pdfs = []
    start = time.perf_counter()

    for pdf_path in tqdm(pdf_files, desc="Converting PDFs", unit="pdf"):
        t0 = time.perf_counter()
        try:
            written = convert_pdf_to_images(pdf_path, output_dir, dpi, skip_existing)
            elapsed = time.perf_counter() - t0
            total_pages += len(written)
            if written:
                log.info(
                    "  %-30s → %2d page(s) written  [%.2fs]",
                    pdf_path.name, len(written), elapsed
                )
            else:
                log.info("  %-30s → all pages already exist, skipped", pdf_path.name)

        except (ValueError, RuntimeError) as e:
            log.error("  FAILED: %s — %s", pdf_path.name, e)
            skipped_pdfs.append(pdf_path.name)

    total_elapsed = time.perf_counter() - start
    log.info(
        "Done. %d page(s) written from %d PDF(s) in %.2fs",
        total_pages, len(pdf_files) - len(skipped_pdfs), total_elapsed
    )
    if skipped_pdfs:
        log.warning("Skipped PDFs (check above for errors): %s", skipped_pdfs)

    return {
        "total_pdfs"  : len(pdf_files),
        "total_pages" : total_pages,
        "skipped_pdfs": skipped_pdfs,
        "output_dir"  : str(output_dir),
        "elapsed_sec" : round(total_elapsed, 2),
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    summary = convert_all_pdfs()
    print("\n── Summary ──────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")
    print("─────────────────────────────────────────")