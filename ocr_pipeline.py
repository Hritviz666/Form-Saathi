

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import rotate as scipy_rotate

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Tiling
TILE_MAX_HEIGHT    = 1400    # px — tiles taller than this get split further
TILE_OVERLAP_PX    = 150     # px — overlap between consecutive tiles (avoids cutting words)
MIN_CONTENT_ROWS   = 20      # px — minimum content band height to keep as a tile

# Parallel OCR
MAX_WORKERS        = 4       # ThreadPoolExecutor threads for tile OCR
BATCH_SIZE         = 8       # max tiles per Surya batch call

# OCR languages — covers Indian government forms
OCR_LANGS          = ["en", "hi"]

# Deduplication
IOU_THRESHOLD      = 0.4     # boxes with IoU > this are considered duplicates

# Preprocessing
DESKEW_ANGLE_LIMIT = 15.0    # degrees — ignore larger angles (likely not skew)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ocr_pipeline")


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class TextLine:
    """Single OCR-detected text line with position and confidence."""
    text       : str
    bbox       : list[float]   # [x1, y1, x2, y2] in full-image pixel coordinates
    confidence : float
    tile_idx   : int = -1      # which tile this came from (-1 = full image)


@dataclass
class OCRResult:
    """Full OCR output for one form image."""
    image_path   : str
    lines        : list[TextLine] = field(default_factory=list)
    elapsed_sec  : float = 0.0
    tile_count   : int   = 0
    skew_angle   : float = 0.0

    @property
    def full_text(self) -> str:
        """All detected text joined by newlines, in top-to-bottom order."""
        sorted_lines = sorted(self.lines, key=lambda l: (l.bbox[1], l.bbox[0]))
        return "\n".join(l.text for l in sorted_lines if l.text.strip())

    @property
    def as_dict(self) -> dict:
        return {
            "image_path" : self.image_path,
            "elapsed_sec": self.elapsed_sec,
            "tile_count" : self.tile_count,
            "skew_angle" : self.skew_angle,
            "line_count" : len(self.lines),
            "lines"      : [
                {
                    "text"      : l.text,
                    "bbox"      : l.bbox,
                    "confidence": l.confidence,
                }
                for l in self.lines
            ],
        }


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def deskew(img_gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Estimate and correct skew angle using Hough line transform.

    Args:
        img_gray: Grayscale image (H, W) uint8

    Returns:
        (deskewed_image, angle_degrees)
        angle is 0.0 if skew is outside DESKEW_ANGLE_LIMIT or undetectable.
    """
    # edge detection
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)

    # Hough lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )

    if lines is None:
        return img_gray, 0.0

    # compute angle of each line
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    if not angles:
        return img_gray, 0.0

    median_angle = float(np.median(angles))

    # ignore implausible angles
    if abs(median_angle) > DESKEW_ANGLE_LIMIT:
        return img_gray, 0.0

    # rotate to correct skew
    corrected = scipy_rotate(
        img_gray,
        angle=-median_angle,
        reshape=False,
        cval=255,       # white fill
        order=1,        # bilinear
    ).astype(np.uint8)

    return corrected, median_angle


def denoise(img_gray: np.ndarray) -> np.ndarray:
    """
    Denoise using Non-local Means — handles scanner grain and JPEG artifacts
    without blurring text strokes (unlike Gaussian blur).
    """
    return cv2.fastNlMeansDenoising(
        img_gray,
        h=10,               # filter strength (10 is conservative, preserves thin strokes)
        templateWindowSize=7,
        searchWindowSize=21,
    )


def preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Full preprocessing pipeline: grayscale → denoise → deskew.

    Returns:
        (preprocessed_gray_image, skew_angle)
    """
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised  = denoise(gray)
    deskewed, angle = deskew(denoised)
    return deskewed, angle


# ─────────────────────────────────────────────
# ADAPTIVE TILING
# ─────────────────────────────────────────────

def find_content_rows(img_gray: np.ndarray, threshold: int = 240) -> np.ndarray:
    """
    For each row, return True if the row contains content (non-white pixels).

    Args:
        threshold: pixels darker than this are considered content

    Returns:
        Boolean array of shape (H,)
    """
    return np.any(img_gray < threshold, axis=1)


def find_split_points(
    content_rows: np.ndarray,
    max_tile_height: int = TILE_MAX_HEIGHT,
    min_content_rows: int = MIN_CONTENT_ROWS,
) -> list[tuple[int, int]]:
    """
    Find horizontal split points that fall inside content-sparse bands.

    Strategy:
        - Scan top to bottom
        - When accumulated height exceeds max_tile_height, look ahead for
          the nearest empty band (no content rows) to split cleanly
        - If no empty band found within a lookahead window, split at the
          max_tile_height boundary regardless

    Returns:
        List of (start_row, end_row) tuples for each tile.
    """
    H = len(content_rows)
    tiles = []
    start = 0

    while start < H:
        end_ideal = start + max_tile_height

        if end_ideal >= H:
            # last tile — take the rest
            tiles.append((start, H))
            break

        # look for an empty band in the ±200px window around end_ideal
        search_start = max(start + min_content_rows, end_ideal - 200)
        search_end   = min(H - 1, end_ideal + 200)

        split_at = end_ideal  # fallback

        # find longest consecutive empty stretch near end_ideal
        best_len   = 0
        best_mid   = end_ideal
        run_start  = None

        for r in range(search_start, search_end):
            if not content_rows[r]:
                if run_start is None:
                    run_start = r
                run_len = r - run_start + 1
                if run_len > best_len:
                    best_len = run_len
                    best_mid = run_start + run_len // 2
            else:
                run_start = None

        if best_len >= 5:
            split_at = best_mid

        tiles.append((start, split_at))
        # next tile starts with overlap
        start = max(start + 1, split_at - TILE_OVERLAP_PX)

    return tiles


def extract_tiles(
    img_gray: np.ndarray,
    tile_bounds: list[tuple[int, int]],
) -> list[tuple[np.ndarray, int]]:
    """
    Extract image tiles from (start_row, end_row) bounds.

    Returns:
        List of (tile_image, y_offset) where y_offset is the start_row
        in full-image coordinates (used to translate bbox back).
    """
    tiles = []
    for y1, y2 in tile_bounds:
        tile = img_gray[y1:y2, :]
        tiles.append((tile, y1))
    return tiles


def adaptive_tile(img_gray: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """
    Full adaptive tiling pipeline for one image.

    Returns list of (tile_array, y_offset_in_full_image).
    If image height ≤ TILE_MAX_HEIGHT, returns the full image as one tile.
    """
    H = img_gray.shape[0]

    if H <= TILE_MAX_HEIGHT:
        return [(img_gray, 0)]

    content_rows = find_content_rows(img_gray)
    tile_bounds  = find_split_points(content_rows)
    return extract_tiles(img_gray, tile_bounds)


# ─────────────────────────────────────────────
# BOUNDING BOX UTILITIES
# ─────────────────────────────────────────────

def surya_bbox_to_xyxy(bbox) -> list[float]:
    """
    Convert Surya's bbox format to [x1, y1, x2, y2].
    Surya returns bbox as [x1, y1, x2, y2] already — but handle
    both list and object formats defensively.
    """
    if hasattr(bbox, "bbox"):
        b = bbox.bbox
    elif isinstance(bbox, (list, tuple)):
        b = list(bbox)
    else:
        b = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
    return [float(x) for x in b[:4]]


def translate_bbox(bbox: list[float], y_offset: int) -> list[float]:
    """Shift bbox y-coordinates by tile's y_offset to get full-image coords."""
    x1, y1, x2, y2 = bbox
    return [x1, y1 + y_offset, x2, y2 + y_offset]


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection over Union for two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def nms_dedup(lines: list[TextLine], iou_thresh: float = IOU_THRESHOLD) -> list[TextLine]:
    """
    Remove duplicate TextLines caused by tile overlap using IoU-based NMS.
    Keeps the detection with the higher confidence score.

    Runs in O(n²) — acceptable for typical form line counts (<500 lines).
    """
    if not lines:
        return lines

    # sort by confidence descending
    lines = sorted(lines, key=lambda l: l.confidence, reverse=True)
    kept  = []

    for candidate in lines:
        duplicate = False
        for kept_line in kept:
            iou = compute_iou(candidate.bbox, kept_line.bbox)
            if iou > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    return kept


# ─────────────────────────────────────────────
# SURYA OCR RUNNER
# ─────────────────────────────────────────────

class SuryaEngine:
    """
    Singleton wrapper around Surya OCR models.
    Models are loaded once on first use and reused across all calls.
    Thread-safe: lock prevents multiple threads from loading simultaneously.
    """
    _instance = None
    _lock     = __import__("threading").Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._loaded = False
        return cls._instance

    def load(self):
        with self._lock:
            if self._loaded:
                return

            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype       = torch.float16 if self.device == "cuda" else torch.float32
            log.info("Loading Surya OCR models on %s (dtype: %s)...", self.device, dtype)
            t0 = time.perf_counter()

            import os
            os.environ["RECOGNITION_BATCH_SIZE"] = "256" if self.device == "cuda" else "32"
            os.environ["DETECTOR_BATCH_SIZE"]    = "36"  if self.device == "cuda" else "6"

            from surya.model.detection.segformer import (
                load_model as load_det_model,
                load_processor as load_det_processor,
            )
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor

            self.det_model     = load_det_model()
            self.det_processor = load_det_processor()
            self.rec_model     = load_rec_model()
            self.rec_processor = load_rec_processor()
            self._loaded       = True
            log.info("Surya models loaded in %.2fs on %s", time.perf_counter() - t0, self.device)

    def ocr_batch(
        self,
        pil_images: list,
        langs: list[list[str]] = None,
    ) -> list:
        """
        Run OCR on a batch of PIL Images.

        Args:
            pil_images: list of PIL.Image objects
            langs:      list of lang lists, one per image. Defaults to OCR_LANGS for all.

        Returns:
            List of Surya OCRResult objects.
        """
        from surya.ocr import run_ocr

        self.load()

        if langs is None:
            langs = [OCR_LANGS] * len(pil_images)

        return run_ocr(
            pil_images,
            langs,
            self.det_model,
            self.det_processor,
            self.rec_model,
            self.rec_processor,
        )


# ─────────────────────────────────────────────
# TILE OCR WORKER
# ─────────────────────────────────────────────

def ocr_tile(
    engine: SuryaEngine,
    tile_img: np.ndarray,
    y_offset: int,
    tile_idx: int,
) -> list[TextLine]:
    """
    Run OCR on a single tile and return TextLines with full-image coordinates.

    Args:
        engine:    SuryaEngine singleton
        tile_img:  Grayscale tile as numpy array
        y_offset:  Tile's y position in the full image (for bbox translation)
        tile_idx:  Index of this tile (for debugging)

    Returns:
        List of TextLine objects with full-image bboxes.
    """
    # Surya expects PIL RGB images
    pil_tile = Image.fromarray(tile_img).convert("RGB")

    results  = engine.ocr_batch([pil_tile])
    lines    = []

    if not results or not results[0].text_lines:
        return lines

    for text_line in results[0].text_lines:
        raw_bbox   = surya_bbox_to_xyxy(text_line)
        full_bbox  = translate_bbox(raw_bbox, y_offset)
        confidence = float(getattr(text_line, "confidence", 1.0))
        text       = getattr(text_line, "text", "").strip()

        if not text:
            continue

        lines.append(TextLine(
            text       = text,
            bbox       = full_bbox,
            confidence = confidence,
            tile_idx   = tile_idx,
        ))

    return lines


# ─────────────────────────────────────────────
# MAIN OCR CLASS
# ─────────────────────────────────────────────

class FormOCR:
    """
    Main OCR interface for FormSaathi.

    Usage:
        ocr = FormOCR()

        # from file path
        result = ocr.run("data/forms/images/Form49A/Form49A_page001.png")

        # from numpy array (BGR, as returned by cv2.imread)
        result = ocr.run(img_array)

        # access results
        print(result.full_text)
        for line in result.lines:
            print(line.text, line.bbox, line.confidence)
    """

    def __init__(self, max_workers: int = MAX_WORKERS):
        self.engine      = SuryaEngine()
        self.max_workers = max_workers

    def run(
        self,
        image: str | Path | np.ndarray,
        langs: list[str] = None,
    ) -> OCRResult:
        """
        Full OCR pipeline on a single form image.

        Args:
            image: File path (str/Path) or BGR numpy array from cv2.imread
            langs: OCR language codes. Defaults to ["en", "hi"].

        Returns:
            OCRResult with all detected text lines and metadata.
        """
        t_start = time.perf_counter()

        # ── load image ─────────────────────────────
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img_bgr    = cv2.imread(image_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot load image: {image_path}")
        else:
            img_bgr    = image
            image_path = "<array>"

        # ── preprocess ─────────────────────────────
        img_gray, skew_angle = preprocess(img_bgr)
        log.debug("Preprocessed — skew: %.2f°", skew_angle)

        # ── adaptive tiling ────────────────────────
        tiles = adaptive_tile(img_gray)
        log.debug("Tiles: %d", len(tiles))

        # ── batch OCR — all tiles in one Surya call ─
        # Threading hurts GPU performance: PyTorch serializes CUDA ops anyway.
        # Batching all tiles into one forward pass is 2-3× faster.
        all_lines: list[TextLine] = []

        pil_tiles  = [Image.fromarray(t).convert("RGB") for t, _ in tiles]
        y_offsets  = [y for _, y in tiles]
        batch_langs = [langs or OCR_LANGS] * len(pil_tiles)

        try:
            self.engine.load()
            surya_results = self.engine.ocr_batch(pil_tiles, batch_langs)

            for tile_idx, (result, y_off) in enumerate(zip(surya_results, y_offsets)):
                if not result.text_lines:
                    continue
                for text_line in result.text_lines:
                    raw_bbox   = surya_bbox_to_xyxy(text_line)
                    full_bbox  = translate_bbox(raw_bbox, y_off)
                    confidence = float(getattr(text_line, "confidence", 1.0))
                    text       = getattr(text_line, "text", "").strip()
                    if not text:
                        continue
                    all_lines.append(TextLine(
                        text       = text,
                        bbox       = full_bbox,
                        confidence = confidence,
                        tile_idx   = tile_idx,
                    ))
        except Exception as e:
            log.error("Batch OCR failed: %s", e)
            raise

        # ── deduplicate overlap regions ─────────────
        if len(tiles) > 1:
            before = len(all_lines)
            all_lines = nms_dedup(all_lines)
            log.debug("NMS: %d → %d lines", before, len(all_lines))

        # ── sort top-to-bottom, left-to-right ──────
        all_lines.sort(key=lambda l: (round(l.bbox[1] / 20), l.bbox[0]))

        elapsed = time.perf_counter() - t_start

        log.info(
            "OCR done — %d lines | %d tiles | %.2fs | skew: %.1f° | %s",
            len(all_lines), len(tiles), elapsed,
            skew_angle, Path(image_path).name,
        )

        return OCRResult(
            image_path  = image_path,
            lines       = all_lines,
            elapsed_sec = round(elapsed, 3),
            tile_count  = len(tiles),
            skew_angle  = round(skew_angle, 2),
        )

    def run_batch(
        self,
        images: list,
        langs: list[str] = None,
    ) -> list[OCRResult]:
        """
        Run OCR on a list of images (paths or arrays).
        Processes sequentially — each image already uses internal parallelism.
        """
        results = []
        for img in images:
            try:
                results.append(self.run(img, langs))
            except Exception as e:
                log.error("Batch OCR failed on %s: %s", img, e)
        return results


# ─────────────────────────────────────────────
# ENTRY POINT — quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_pipeline.py <image_path>")
        print("Example: python ocr_pipeline.py data/forms/images/Form49A/Form49A_page001.png")
        sys.exit(0)

    image_path = sys.argv[1]
    log.info("Running OCR on: %s", image_path)

    ocr    = FormOCR()
    result = ocr.run(image_path)

    print("\n── OCR Result ───────────────────────────────────────")
    print(f"  Image      : {result.image_path}")
    print(f"  Lines      : {len(result.lines)}")
    print(f"  Tiles used : {result.tile_count}")
    print(f"  Skew angle : {result.skew_angle}°")
    print(f"  Elapsed    : {result.elapsed_sec}s")
    print("\n── Extracted Text ───────────────────────────────────")
    print(result.full_text)
    print("─────────────────────────────────────────────────────")