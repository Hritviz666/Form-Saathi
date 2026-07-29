

from ultralytics import YOLO
import yaml, torch
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_YAML = "data/yolo/dataset.yaml"
BASE_MODEL   = "yolov8n.pt"   # restart clean — don't resume last.pt (size changed)
OUTPUT_NAME  = "formfields"   # saved under runs/detect/formfields/

EPOCHS        = 50
IMG_SIZE      = 640    # OOM fix: 1280 exceeded 8GB VRAM; 640 is safe at ~4.5-5.5GB
BATCH         = 8      # safe for 8GB at IMG_SIZE=640
PATIENCE      = 10     # early stop if no mAP improvement for 10 epochs
WORKERS       = 4

CLASS_NAMES = {
    0: "text_field",
    1: "checkbox",
    2: "date_field",
    3: "signature_box",
    4: "dropdown",
}

# ── Sanity-check dataset.yaml ─────────────────────────────────────────────────

def verify_dataset(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    required_keys = {"path", "train", "val", "names"}
    missing = required_keys - set(cfg.keys())
    if missing:
        raise ValueError(f"dataset.yaml missing keys: {missing}")

    train_img_dir = Path(cfg["path"]) / cfg["train"]
    val_img_dir   = Path(cfg["path"]) / cfg["val"]

    train_count = len(list(train_img_dir.glob("*.png"))) + \
                  len(list(train_img_dir.glob("*.jpg")))
    val_count   = len(list(val_img_dir.glob("*.png"))) + \
                  len(list(val_img_dir.glob("*.jpg")))

    print(f"[dataset]  train images : {train_count}")
    print(f"[dataset]  val   images : {val_count}")
    print(f"[dataset]  classes      : {cfg['names']}")

    if train_count == 0:
        raise FileNotFoundError(f"No train images found at {train_img_dir}")
    if val_count == 0:
        raise FileNotFoundError(f"No val images found at {val_img_dir}")

    return cfg


# ── VRAM check before training ────────────────────────────────────────────────

def check_vram():
    """Warn if available VRAM looks too low even for the safe config."""
    if not torch.cuda.is_available():
        print("[warning]  CUDA not available — training will run on CPU (very slow)")
        return

    total_vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_vram_gb   = (torch.cuda.get_device_properties(0).total_memory
                      - torch.cuda.memory_allocated(0)) / 1e9

    print(f"[torch]  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"[torch]  Total VRAM     : {total_vram_gb:.1f} GB")
    print(f"[torch]  Free  VRAM     : {free_vram_gb:.1f} GB")

    if free_vram_gb < 4.0:
        print("[warning]  Less than 4GB free — consider closing other GPU processes.")
        print("[warning]  If OOM occurs again, reduce BATCH from 8 to 4.")


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> Path:
    check_vram()
    verify_dataset(DATASET_YAML)

    print(f"\n[config]  IMG_SIZE={IMG_SIZE}  BATCH={BATCH}  EPOCHS={EPOCHS}  PATIENCE={PATIENCE}")
    print(f"[config]  amp=True (float16 activations — halves VRAM usage)")
    print(f"[config]  cache=ram (pre-loads resized images — faster epoch iteration)\n")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data    = DATASET_YAML,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        patience= PATIENCE,
        workers = WORKERS,
        name    = OUTPUT_NAME,
        project = "runs/detect",

        # ── OOM fixes ────────────────────────────────────────────────────────
        amp     = True,       # mixed precision (float16) — halves activation memory
        cache   = "ram",      # cache resized images in RAM — avoids repeated resize I/O

        # ── Augmentation (light — forms have fixed orientation post-deskew) ──
        hsv_h       = 0.01,   # minimal hue shift
        hsv_s       = 0.3,    # saturation variation (scan quality differs)
        hsv_v       = 0.3,    # brightness variation (scan quality differs)
        degrees     = 2.0,    # slight rotation only (deskew handles large angles)
        translate   = 0.05,   # minor translation
        scale       = 0.3,    # zoom variation
        shear       = 0.0,    # no shear — forms don't warp
        flipud      = 0.0,    # no vertical flip — forms always upright post-deskew
        fliplr      = 0.0,    # no horizontal flip — forms not symmetric

        # ── Optimizer ────────────────────────────────────────────────────────
        optimizer     = "AdamW",
        lr0           = 1e-3,
        lrf           = 0.01,   # final LR = lr0 * lrf = 1e-5
        weight_decay  = 5e-4,
        warmup_epochs = 5,
        cos_lr        = True,   # cosine LR schedule — smooth decay

        # ── Output ───────────────────────────────────────────────────────────
        save_period = 10,     # save checkpoint every 10 epochs
        plots       = True,   # save training curves to runs/detect/formfields/
        verbose     = True,
    )

    best_path = Path("runs/detect") / OUTPUT_NAME / "weights" / "best.pt"

    print(f"\n{'='*60}")
    print(f"[done]  Best weights → {best_path}")

    map50    = results.results_dict.get("metrics/mAP50(B)",    "N/A")
    map5095  = results.results_dict.get("metrics/mAP50-95(B)", "N/A")

    if isinstance(map50, float):
        print(f"[metrics]  mAP@0.5       : {map50:.4f}")
        print(f"[metrics]  mAP@0.5:0.95  : {map5095:.4f}")
        if map50 < 0.75:
            print("[warning]  mAP@0.5 below 0.75 — consider:")
            print("           • Checking label quality in data/yolo/labels/")
            print("           • Increasing EPOCHS to 80")
            print("           • Bumping IMG_SIZE to 800 if VRAM allows")
    else:
        print(f"[metrics]  mAP@0.5       : {map50}")
        print(f"[metrics]  mAP@0.5:0.95  : {map5095}")

    print(f"{'='*60}\n")
    return best_path


# ── Quick inference test ──────────────────────────────────────────────────────

def quick_test(weights_path: str, test_image: str):
    """
    Sanity-check: run best.pt on one form image and print all detections.

    Usage (after training completes):
        from train_yolo import quick_test
        quick_test(
            "runs/detect/formfields/weights/best.pt",
            "data/forms/images/Form49A/page_1.png"
        )
    """
    model = YOLO(weights_path)
    results = model.predict(
        test_image,
        imgsz   = IMG_SIZE,
        conf    = 0.3,    # low threshold — merger will re-filter
        iou     = 0.4,
        verbose = False,
    )

    for r in results:
        print(f"\n[test]  {len(r.boxes)} fields detected in {test_image}")
        print(f"{'Class':<16}  {'Conf':>6}  BBox [x1, y1, x2, y2]")
        print("-" * 60)
        for box in sorted(r.boxes, key=lambda b: float(b.xyxy[0][1])):  # sort top→bottom
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [round(v) for v in box.xyxy[0].tolist()]
            print(f"  {CLASS_NAMES[cls]:<14}  {conf:>6.2f}  {xyxy}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    best = train()
    print("Training complete.")
    print(f"Next step: run field_merger.py with weights at {best}")
    print("Or run quick_test() on a sample form to verify detections.")