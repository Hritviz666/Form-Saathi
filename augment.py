

import os
import cv2
import json
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Pipeline A — form images (OCR training)
FORMS_INPUT_DIR      = "data/forms/images"       # subfolders per form PDF
FORMS_OUTPUT_DIR     = "data/augmented/forms"    # flat output, all augmented images here
FORMS_TARGET_TOTAL   = 600                        # minimum total augmented images to generate
AUG_PER_IMAGE        = 20                         # augmented versions per source image

# Pipeline B — YOLO images (field detection training)
YOLO_IMG_INPUT_DIR   = "data/yolo/images/train"
YOLO_LBL_INPUT_DIR   = "data/yolo/labels/train"
YOLO_IMG_OUTPUT_DIR  = "data/augmented/yolo/images/train"
YOLO_LBL_OUTPUT_DIR  = "data/augmented/yolo/labels/train"
YOLO_AUG_FACTOR      = 4       # how many augmented copies per YOLO image

SKIP_EXISTING        = True
SEED                 = 42

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("augment")
random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# AUGMENTATION PIPELINES
# ─────────────────────────────────────────────

def build_light_transform() -> A.Compose:
    """
    Light augmentation — simulates clean scanner output with minor variations.
    Safe for both OCR and YOLO pipelines.
    """
    return A.Compose([
        A.Rotate(
            limit=5,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,          # white border (form background)
            p=0.7
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.8
        ),
        A.GaussNoise(
            var_limit=(10.0, 40.0),
            mean=0,
            p=0.5
        ),
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.3
        ),
        A.Sharpen(
            alpha=(0.1, 0.3),
            lightness=(0.9, 1.1),
            p=0.3
        ),
    ])


def build_heavy_transform() -> A.Compose:
    """
    Heavy augmentation — simulates crumpled paper, camera photos, bad scans.
    Covers real-world conditions for Indian government form submissions.
    """
    return A.Compose([
        A.Rotate(
            limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
            p=0.8
        ),
        A.Perspective(
            scale=(0.02, 0.06),
            keep_size=True,
            p=0.4
        ),
        A.ElasticTransform(
            alpha=40,
            sigma=5,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
            p=0.35
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.15,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
            p=0.3
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.35,
            contrast_limit=0.35,
            p=0.9
        ),
        A.GaussNoise(
            var_limit=(20.0, 80.0),
            mean=0,
            p=0.6
        ),
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=0.45
        ),
        A.ImageCompression(
            quality_lower=45,
            quality_upper=85,
            p=0.5
        ),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=4,
            p=0.2
        ),
        A.ToGray(p=0.15),     # some scans come out grayscale
    ])


def build_yolo_light_transform() -> A.Compose:
    """Light transform with bbox support for YOLO pipeline."""
    return A.Compose(
        build_light_transform().transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,     # drop boxes that become <30% visible after crop/rotate
            clip=True,
        )
    )


def build_yolo_heavy_transform() -> A.Compose:
    """Heavy transform with bbox support for YOLO pipeline."""
    return A.Compose(
        build_heavy_transform().transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
            clip=True,
        )
    )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_image_cv2(path: Path) -> np.ndarray:
    """Load image as BGR numpy array via OpenCV."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def save_image_cv2(img: np.ndarray, path: Path) -> None:
    """Save BGR numpy array as PNG."""
    cv2.imwrite(str(path), img)


def read_yolo_labels(label_path: Path) -> tuple[list[int], list[list[float]]]:
    """
    Read a YOLOv8 .txt label file.

    Returns:
        class_ids : list of int
        bboxes    : list of [cx, cy, w, h] (normalized floats)
    """
    class_ids = []
    bboxes    = []

    if not label_path.exists() or label_path.stat().st_size == 0:
        return class_ids, bboxes

    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_ids.append(int(parts[0]))
        bboxes.append([float(x) for x in parts[1:]])

    return class_ids, bboxes


def write_yolo_labels(
    class_ids: list[int],
    bboxes: list,
    out_path: Path
) -> None:
    """Write YOLOv8 .txt label file from class IDs and bboxes."""
    lines = []
    for cid, box in zip(class_ids, bboxes):
        cx, cy, w, h = box
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def pick_transform(light_t, heavy_t, heavy_prob: float = 0.4):
    """Return light or heavy transform based on probability."""
    return heavy_t if random.random() < heavy_prob else light_t


# ─────────────────────────────────────────────
# PIPELINE A — Form image augmentation (OCR)
# ─────────────────────────────────────────────

def augment_form_images(
    input_dir: str | Path  = FORMS_INPUT_DIR,
    output_dir: str | Path = FORMS_OUTPUT_DIR,
    aug_per_image: int     = AUG_PER_IMAGE,
    skip_existing: bool    = SKIP_EXISTING,
) -> dict:
    """
    Augment PDF-converted form images for OCR training.
    No bounding box tracking needed — OCR trains on image+text pairs.

    Walks all subfolders under input_dir, collects all PNGs,
    generates aug_per_image augmented versions of each.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    light_t = build_light_transform()
    heavy_t = build_heavy_transform()

    # collect all source images across all form subfolders
    source_images = sorted(input_dir.rglob("*.png"))
    if not source_images:
        log.warning("No PNG images found under: %s", input_dir)
        return {}

    log.info("Pipeline A — Form image augmentation")
    log.info("  Source images : %d", len(source_images))
    log.info("  Aug per image : %d", aug_per_image)
    log.info("  Expected total: ~%d", len(source_images) * aug_per_image)

    written  = 0
    skipped  = 0
    failed   = 0

    for img_path in tqdm(source_images, desc="Augmenting forms", unit="img"):
        try:
            img = load_image_cv2(img_path)
        except RuntimeError as e:
            log.error("  %s", e)
            failed += 1
            continue

        stem = img_path.stem   # e.g. "Form49A_page001"

        for i in range(aug_per_image):
            out_name = f"{stem}_aug{i:03d}.png"
            out_path = output_dir / out_name

            if skip_existing and out_path.exists():
                skipped += 1
                continue

            transform = pick_transform(light_t, heavy_t)
            result    = transform(image=img)
            augmented = result["image"]

            save_image_cv2(augmented, out_path)
            written += 1

    log.info(
        "  Done — written: %d | skipped: %d | failed: %d",
        written, skipped, failed
    )
    return {
        "source_images": len(source_images),
        "written"      : written,
        "skipped"      : skipped,
        "failed"       : failed,
    }


# ─────────────────────────────────────────────
# PIPELINE B — YOLO image augmentation (bbox-aware)
# ─────────────────────────────────────────────

def augment_yolo_images(
    img_input_dir:  str | Path = YOLO_IMG_INPUT_DIR,
    lbl_input_dir:  str | Path = YOLO_LBL_INPUT_DIR,
    img_output_dir: str | Path = YOLO_IMG_OUTPUT_DIR,
    lbl_output_dir: str | Path = YOLO_LBL_OUTPUT_DIR,
    aug_factor:     int        = YOLO_AUG_FACTOR,
    skip_existing:  bool       = SKIP_EXISTING,
) -> dict:
    """
    Augment YOLO training images with corresponding bounding box label updates.
    Bboxes are transformed in sync with the image — no manual coordinate math needed.

    Output goes to a separate augmented folder so original FUNSD data is untouched.
    Copy or symlink into data/yolo/images/train when ready for training.
    """
    img_input_dir  = Path(img_input_dir)
    lbl_input_dir  = Path(lbl_input_dir)
    img_output_dir = Path(img_output_dir)
    lbl_output_dir = Path(lbl_output_dir)

    img_output_dir.mkdir(parents=True, exist_ok=True)
    lbl_output_dir.mkdir(parents=True, exist_ok=True)

    light_t = build_yolo_light_transform()
    heavy_t = build_yolo_heavy_transform()

    source_images = sorted(img_input_dir.glob("*.png"))
    if not source_images:
        log.warning("No PNG images found in: %s", img_input_dir)
        return {}

    log.info("Pipeline B — YOLO image augmentation (bbox-aware)")
    log.info("  Source images : %d", len(source_images))
    log.info("  Aug factor    : %d×", aug_factor)
    log.info("  Expected total: ~%d new samples", len(source_images) * aug_factor)

    written      = 0
    skipped      = 0
    failed       = 0
    empty_labels = 0   # images with no annotations (valid — YOLOv8 uses negatives)

    for img_path in tqdm(source_images, desc="Augmenting YOLO", unit="img"):
        stem      = img_path.stem
        lbl_path  = lbl_input_dir / f"{stem}.txt"

        try:
            img = load_image_cv2(img_path)
        except RuntimeError as e:
            log.error("  %s", e)
            failed += 1
            continue

        class_ids, bboxes = read_yolo_labels(lbl_path)

        if not bboxes:
            empty_labels += 1

        for i in range(aug_factor):
            out_img_name = f"{stem}_aug{i:03d}.png"
            out_lbl_name = f"{stem}_aug{i:03d}.txt"
            out_img_path = img_output_dir / out_img_name
            out_lbl_path = lbl_output_dir / out_lbl_name

            if skip_existing and out_img_path.exists():
                skipped += 1
                continue

            transform = pick_transform(light_t, heavy_t)

            try:
                result = transform(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_ids,
                )
            except Exception as e:
                log.warning("  Transform failed on %s aug%d: %s", stem, i, e)
                failed += 1
                continue

            aug_img    = result["image"]
            aug_bboxes = result["bboxes"]
            aug_labels = result["class_labels"]

            save_image_cv2(aug_img, out_img_path)
            write_yolo_labels(aug_labels, aug_bboxes, out_lbl_path)
            written += 1

    log.info(
        "  Done — written: %d | skipped: %d | failed: %d | empty-label imgs: %d",
        written, skipped, failed, empty_labels
    )
    return {
        "source_images": len(source_images),
        "written"      : written,
        "skipped"      : skipped,
        "failed"       : failed,
        "empty_labels" : empty_labels,
    }


# ─────────────────────────────────────────────
# MERGE HELPER — copy augmented YOLO data into main train folder
# ─────────────────────────────────────────────

def merge_augmented_into_train(
    aug_img_dir: str | Path = YOLO_IMG_OUTPUT_DIR,
    aug_lbl_dir: str | Path = YOLO_LBL_OUTPUT_DIR,
    train_img_dir: str | Path = YOLO_IMG_INPUT_DIR,
    train_lbl_dir: str | Path = YOLO_LBL_INPUT_DIR,
) -> None:
    """
    Copy augmented YOLO images + labels into the main train folder.
    Call this after reviewing augmented output and confirming quality.

    Safe to call multiple times — skips files that already exist in train/.
    """
    import shutil

    aug_img_dir   = Path(aug_img_dir)
    aug_lbl_dir   = Path(aug_lbl_dir)
    train_img_dir = Path(train_img_dir)
    train_lbl_dir = Path(train_lbl_dir)

    copied = 0
    for img_path in aug_img_dir.glob("*.png"):
        dst = train_img_dir / img_path.name
        if not dst.exists():
            shutil.copy2(img_path, dst)
            copied += 1

    for lbl_path in aug_lbl_dir.glob("*.txt"):
        dst = train_lbl_dir / lbl_path.name
        if not dst.exists():
            shutil.copy2(lbl_path, dst)

    log.info("Merged %d augmented YOLO samples into train/", copied)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_augmentation():
    log.info("═" * 52)
    log.info("FormSaathi — Augmentation Pipeline")
    log.info("═" * 52)

    stats_a = augment_form_images()
    log.info("")
    stats_b = augment_yolo_images()

    log.info("")
    log.info("═" * 52)
    log.info("All done.")
    log.info(
        "Pipeline A: %d form images → %d augmented OCR images",
        stats_a.get("source_images", 0),
        stats_a.get("written", 0),
    )
    log.info(
        "Pipeline B: %d YOLO images → %d augmented YOLO images",
        stats_b.get("source_images", 0),
        stats_b.get("written", 0),
    )
    log.info("")
    log.info("Next step:")
    log.info("  1. Visually inspect a few images in data/augmented/")
    log.info("  2. Call merge_augmented_into_train() to add to YOLO train set")
    log.info("  3. Move to OCR pipeline (adaptive tiling)")
    log.info("═" * 52)


if __name__ == "__main__":
    run_augmentation()