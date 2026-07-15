

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image as PILImage
from ultralytics import YOLO

from ocr_pipeline import SuryaEngine
from surya.schema import OCRResult, TextLine

# ── Config ────────────────────────────────────────────────────────────────────

YOLO_WEIGHTS = "D:/FormSaathi/runs/detect/formfields4/weights/best.pt"
YOLO_CONF       = 0.40    # raised from 0.30 — reduces false detections
YOLO_IOU        = 0.40    # NMS IoU threshold
YOLO_IMG_SIZE   = 640
MIN_FIELD_AREA  = 8000    # px² — filters per-character input cells (PAN boxes etc.)

OCR_IOU_THRESHOLD = 0.10  # min IoU for OCR line → YOLO box assignment
CENTROID_ASSIGN   = True  # centroid-inside-box takes priority over IoU
MAX_ORPHAN_DIST   = 80    # px — max distance for orphan snap to nearest field
MIN_OCR_CONF      = 0.40  # discard low-confidence OCR lines

CLASS_NAMES = {
    0: "text_field",
    1: "checkbox",
    2: "date_field",
    3: "signature_box",
    4: "dropdown",
}

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Field:
    """
    A single detected form field with all associated metadata.

    Attributes
    ----------
    field_id   : unique index within the page (top→bottom, left→right order)
    class_name : text_field / checkbox / date_field / signature_box / dropdown
    class_id   : YOLO class integer (0–4)
    bbox       : [x1, y1, x2, y2] in original image pixel coordinates
    conf       : YOLO detection confidence
    ocr_text   : text found inside the YOLO box (joined, cleaned)
    label_text : caption text near the field (above or left of box)
    page       : page number this field was detected on (0-indexed)
    """
    field_id   : int
    class_name : str
    class_id   : int
    bbox       : list[int]
    conf       : float
    ocr_text   : str
    label_text : str
    page       : int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"Field(id={self.field_id}, class={self.class_name}, "
            f"conf={self.conf:.2f}, label={self.label_text!r}, "
            f"text={self.ocr_text!r})"
        )


@dataclass
class FieldList:
    """All fields detected on a single form page."""
    page       : int
    image_path : str
    fields     : list[Field] = field(default_factory=list)
    raw_ocr    : list[dict]  = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.fields)

    def by_class(self, class_name: str) -> list[Field]:
        return [f for f in self.fields if f.class_name == class_name]

    def to_dict(self) -> dict:
        return {
            "page"       : self.page,
            "image_path" : self.image_path,
            "fields"     : [f.to_dict() for f in self.fields],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> str:
        lines = [f"FieldList — page {self.page} — {len(self.fields)} fields"]
        for cls_id, cls_name in CLASS_NAMES.items():
            count = sum(1 for f in self.fields if f.class_id == cls_id)
            if count:
                lines.append(f"  {cls_name:<16} : {count}")
        return "\n".join(lines)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _iou(a: list[int], b: list[int]) -> float:
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(1, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(1, (b[2]-b[0]) * (b[3]-b[1]))
    return inter / (area_a + area_b - inter)


def _centroid(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _centroid_inside(point: tuple[float, float], box: list[int]) -> bool:
    cx, cy = point
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _dist_to_box(point: tuple[float, float], box: list[int]) -> float:
    """Euclidean distance from point to nearest edge of box (0 if inside)."""
    cx, cy = point
    dx = max(box[0] - cx, 0, cx - box[2])
    dy = max(box[1] - cy, 0, cy - box[3])
    return float(np.sqrt(dx*dx + dy*dy))


# ── YOLO detection ────────────────────────────────────────────────────────────

@dataclass
class DetectedField:
    class_id   : int
    class_name : str
    bbox       : list[int]   # [x1,y1,x2,y2] in original image coords
    conf       : float


def run_yolo(image_path: str, weights: str = YOLO_WEIGHTS) -> list[DetectedField]:
    """
    Run YOLOv8 on a form image and return filtered detections.
    Ultralytics predict() auto-rescales bboxes to original image coords.
    MIN_FIELD_AREA filter removes per-character input cells.
    """
    model  = YOLO(weights)
    result = model.predict(
        image_path,
        imgsz   = YOLO_IMG_SIZE,
        conf    = YOLO_CONF,
        iou     = YOLO_IOU,
        verbose = False,
    )[0]

    detections = []
    for box in result.boxes:
        cls_id      = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area        = (x2 - x1) * (y2 - y1)
        if area < MIN_FIELD_AREA:
            continue    # skip character-cell-sized boxes
        detections.append(DetectedField(
            class_id   = cls_id,
            class_name = CLASS_NAMES.get(cls_id, "unknown"),
            bbox       = [round(x1), round(y1), round(x2), round(y2)],
            conf       = round(float(box.conf[0]), 4),
        ))

    # Sort top→bottom, left→right (reading order)
    detections.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
    print(f"[yolo]  {len(detections)} fields detected in {Path(image_path).name}")
    return detections


# ── OCR line filtering ────────────────────────────────────────────────────────

def _filter_ocr_lines(ocr_result: OCRResult) -> list[TextLine]:
    """Remove low-confidence and empty OCR lines."""
    filtered = [
        line for line in ocr_result.text_lines
        if line.confidence >= MIN_OCR_CONF and line.text.strip()
    ]
    print(f"[ocr]   {len(ocr_result.text_lines)} raw lines → {len(filtered)} after filtering")
    return filtered


# ── Core merge logic ──────────────────────────────────────────────────────────

def _assign_ocr_to_fields(
    ocr_lines  : list[TextLine],
    detections : list[DetectedField],
) -> tuple[dict[int, list[TextLine]], list[TextLine]]:
    """
    Assign each OCR line to a YOLO detection box.

    Both OCR and YOLO bboxes are already in original image coordinates
    (Ultralytics auto-rescales; no manual scaling needed).

    Returns
    -------
    assigned : dict mapping detection index → list of OCR lines inside that box
    orphans  : OCR lines not assigned to any box
    """
    det_bboxes = [d.bbox for d in detections]

    assigned : dict[int, list[TextLine]] = {i: [] for i in range(len(detections))}
    orphans  : list[TextLine] = []

    for line in ocr_lines:
        line_bbox     = [int(v) for v in line.bbox]
        line_centroid = _centroid(line_bbox)

        best_idx   : Optional[int] = None
        best_score : float         = -1.0

        for i, det_bbox in enumerate(det_bboxes):
            # Priority 1: centroid inside YOLO box
            if CENTROID_ASSIGN and _centroid_inside(line_centroid, det_bbox):
                score = 1.0 + detections[i].conf
                if score > best_score:
                    best_score = score
                    best_idx   = i
                continue

            # Priority 2: sufficient IoU overlap
            iou = _iou(line_bbox, det_bbox)
            if iou >= OCR_IOU_THRESHOLD and iou > best_score:
                best_score = iou
                best_idx   = i

        if best_idx is not None:
            assigned[best_idx].append(line)
        else:
            orphans.append(line)

    return assigned, orphans


def _snap_orphans(
    orphans    : list[TextLine],
    detections : list[DetectedField],
) -> dict[int, list[TextLine]]:
    """
    Snap orphaned OCR lines to the nearest YOLO box within MAX_ORPHAN_DIST.
    These become label_text candidates (field captions above/beside the box).
    """
    label_lines: dict[int, list[TextLine]] = {i: [] for i in range(len(detections))}
    det_bboxes  = [d.bbox for d in detections]

    for line in orphans:
        line_centroid = _centroid([int(v) for v in line.bbox])
        min_dist = float("inf")
        min_idx  = None

        for i, det_bbox in enumerate(det_bboxes):
            dist = _dist_to_box(line_centroid, det_bbox)
            if dist < min_dist:
                min_dist = dist
                min_idx  = i

        if min_idx is not None and min_dist <= MAX_ORPHAN_DIST:
            label_lines[min_idx].append(line)

    return label_lines


def _sort_lines(lines: list[TextLine]) -> list[TextLine]:
    """Sort OCR lines in reading order: top→bottom, left→right."""
    return sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0]))


def _join_text(lines: list[TextLine]) -> str:
    """Join sorted OCR lines into a single cleaned string."""
    return " ".join(l.text.strip() for l in lines if l.text.strip())


# ── Main merge entry point ────────────────────────────────────────────────────

def merge(
    image_path   : str,
    ocr_result   : OCRResult,
    yolo_weights : str = YOLO_WEIGHTS,
    page         : int = 0,
) -> FieldList:
    """
    Full merge pipeline for one form page.

    Parameters
    ----------
    image_path   : path to the form page image (PNG/JPG)
    ocr_result   : OCRResult from engine.ocr_batch([PIL.Image])[0]
    yolo_weights : path to fine-tuned best.pt
    page         : page index (for multi-page forms)

    Returns
    -------
    FieldList with all detected fields and their OCR text
    """
    # Load image to get original dimensions
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_h, img_w = img.shape[:2]

    # YOLO inference
    # Note: Ultralytics predict() auto-rescales bboxes to original image coords
    detections = run_yolo(image_path, yolo_weights)

    if not detections:
        print(f"[merge]  No YOLO detections on page {page} — returning empty FieldList")
        return FieldList(page=page, image_path=image_path)

    # Filter OCR lines
    ocr_lines = _filter_ocr_lines(ocr_result)

    # Assign OCR lines to YOLO boxes (both in original image coords)
    assigned, orphans = _assign_ocr_to_fields(ocr_lines, detections)
    print(f"[merge]  {len(ocr_lines) - len(orphans)} lines assigned, "
          f"{len(orphans)} orphans → snapping to nearest field")

    # Snap orphans as label candidates
    label_lines = _snap_orphans(orphans, detections)

    # Build FieldList
    fields = []
    for i, det in enumerate(detections):
        inside_lines = _sort_lines(assigned[i])
        nearby_lines = _sort_lines(label_lines[i])

        # Only keep orphan lines that are ABOVE or to the LEFT of the field box
        # (lines below belong to the next field)
        field_top  = det.bbox[1]
        field_left = det.bbox[0]

        caption_lines = [
            l for l in nearby_lines
            if _centroid([int(v) for v in l.bbox])[1] <= field_top  + 10
            or _centroid([int(v) for v in l.bbox])[0] <= field_left + 10
        ]

        fields.append(Field(
            field_id   = i,
            class_name = det.class_name,
            class_id   = det.class_id,
            bbox       = det.bbox,
            conf       = det.conf,
            ocr_text   = _join_text(inside_lines),
            label_text = _join_text(caption_lines),
            page       = page,
        ))

    raw_ocr = [
        {"text": l.text, "bbox": l.bbox, "conf": l.confidence}
        for l in ocr_result.text_lines
    ]

    result = FieldList(
        page       = page,
        image_path = image_path,
        fields     = fields,
        raw_ocr    = raw_ocr,
    )

    print(f"\n{result.summary()}")
    return result


# ── Multi-page helper ─────────────────────────────────────────────────────────

def merge_form(
    image_paths  : list[str],
    yolo_weights : str = YOLO_WEIGHTS,
) -> list[FieldList]:
    """
    Run merge() on every page of a multi-page form.
    SuryaEngine singleton is reused across pages.

    Usage:
        pages = merge_form([
            "data/forms/images/Form49A/Form49A_page001.png",
            "data/forms/images/Form49A/Form49A_page002.png",
        ])
        for page_fields in pages:
            print(page_fields.to_json())
    """
    engine  = SuryaEngine()
    results = []

    for page_idx, img_path in enumerate(image_paths):
        print(f"\n[merge_form]  Processing page {page_idx + 1}/{len(image_paths)}: "
              f"{Path(img_path).name}")
        pil_img    = PILImage.open(img_path).convert("RGB")
        ocr_result = engine.ocr_batch([pil_img])[0]
        field_list = merge(img_path, ocr_result, yolo_weights, page=page_idx)
        results.append(field_list)

    total_fields = sum(len(fl) for fl in results)
    print(f"\n[merge_form]  Done — {len(results)} pages, {total_fields} total fields")
    return results


# ── Debug visualisation ───────────────────────────────────────────────────────

def visualize(
    field_list  : FieldList,
    output_path : str = "debug_merge.png",
) -> None:
    """
    Draw YOLO boxes + OCR text on the form image for visual inspection.
    Saves annotated PNG to output_path.
    """
    CLASS_COLORS = {
        "text_field"    : (0,   200,  50),
        "checkbox"      : (255, 140,   0),
        "date_field"    : (0,   120, 255),
        "signature_box" : (180,   0, 255),
        "dropdown"      : (0,   220, 220),
    }

    img = cv2.imread(field_list.image_path)
    if img is None:
        print(f"[visualize]  Cannot read {field_list.image_path}")
        return

    for f in field_list.fields:
        x1, y1, x2, y2 = f.bbox
        color = CLASS_COLORS.get(f.class_name, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{f.class_name} {f.conf:.2f}"
        if f.label_text:
            label += f" | {f.label_text[:30]}"
        cv2.putText(img, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"[visualize]  Annotated image saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    test_image = "data/forms/images/Form49A/Form49A_page001.png"
    if len(sys.argv) > 1:
        test_image = sys.argv[1]

    if not Path(test_image).exists():
        print(f"[error]  Image not found: {test_image}")
        print("Usage: python field_merger.py <path_to_form_image.png>")
        sys.exit(1)

    print(f"[field_merger]  Running on {test_image}\n")

    engine     = SuryaEngine()
    pil_img    = PILImage.open(test_image).convert("RGB")
    ocr_result = engine.ocr_batch([pil_img])[0]

    field_list = merge(test_image, ocr_result)

    print("\n── Field List ──────────────────────────────────────────")
    for f in field_list.fields:
        print(f"  [{f.field_id:02d}] {f.class_name:<16}"
              f"  conf={f.conf:.2f}"
              f"  label={f.label_text!r:<35}"
              f"  text={f.ocr_text!r}")

    json_out = Path(test_image).stem + "_fields.json"
    Path(json_out).write_text(field_list.to_json(), encoding="utf-8")
    print(f"\n[output]  Field JSON saved → {json_out}")

    visualize(field_list, output_path=Path(test_image).stem + "_debug.png")
    print("\nNext step: run agent.py")