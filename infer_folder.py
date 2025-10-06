# infer_folder.py
import os
import json
import glob
import torch
from ultralytics import YOLO

# -------- User settings (edit these paths) --------
WEIGHTS_PATH = "best.pt"   # path to your trained weights
INPUT_DIR    = "weather_split/test/rainy"  # folder to predict
OUTPUT_JSON  = "infer_results.json"                     # output json path
IMG_SIZE     = 320                                      # inference image size
# --------------------------------------------------

# Select device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# Basic checks
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
if not os.path.isdir(INPUT_DIR):
    raise NotADirectoryError(f"Input directory not found: {INPUT_DIR}")

# Collect images (non-recursive). Add/remove extensions as needed.
exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.heic"]
images = []
for ext in exts:
    images.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
images = sorted({os.path.abspath(p) for p in images})

print(f"Found {len(images)} images in: {INPUT_DIR}")
if not images:
    # Write empty JSON and exit cleanly
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    print(f"No images found. Wrote empty JSON to: {os.path.abspath(OUTPUT_JSON)}")
else:
    # Load classification model
    model = YOLO(WEIGHTS_PATH)

    # Run prediction in a single call; Ultralytics will iterate internally
    results = model.predict(source=images, imgsz=IMG_SIZE, device=device, verbose=False)

    # Build minimal JSON: image absolute path + top-1 class name
    output = []
    for r in results:
        probs = getattr(r, "probs", None)
        names = getattr(r, "names", None) or getattr(model, "names", None)
        img_path = os.path.abspath(getattr(r, "path", ""))

        top1_idx = int(probs.top1)
        pred_name = names[top1_idx]

        output.append({
            "image": img_path,
            "prediction": pred_name
        })

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {os.path.abspath(OUTPUT_JSON)}")