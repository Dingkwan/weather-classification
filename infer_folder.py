# infer_folder.py
import os
import json
import glob
import torch
from ultralytics import YOLO

# -------- User settings (edit these paths) --------
weights_path = "./yolo11_best.pt"   # path to your trained weights
input_dir    = "weather_split/test/rainy"  # folder to predict
output_json  = "infer_results.json"                     # output json path
img_size     = 320                                      # inference image size
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
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights not found: {weights_path}")
if not os.path.isdir(input_dir):
    raise NotADirectoryError(f"Input directory not found: {input_dir}")

# Collect images (non-recursive). Add/remove extensions as needed.
exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.heic"]
images = []
for ext in exts:
    images.extend(glob.glob(os.path.join(input_dir, ext)))
images = sorted({os.path.abspath(p) for p in images})

print(f"Found {len(images)} images in: {input_dir}")
if not images:
    # Write empty JSON and exit cleanly
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    print(f"No images found. Wrote empty JSON to: {os.path.abspath(output_json)}")
else:
    # Load classification model
    model = YOLO(weights_path)

    # Run prediction in a single call; Ultralytics will iterate internally
    results = model.predict(source=images, imgsz=img_size, device=device, verbose=False)

    # Build minimal JSON: image absolute path + top-1 class name
    output = []
    for r in results:
        probs = getattr(r, "probs", None)
        names = getattr(r, "names", None) or getattr(model, "names", None)
        img_path = os.path.abspath(getattr(r, "path", ""))

        top1_idx = int(probs.top1)
        pred_name = names[top1_idx]
        top1_conf = r.probs.top1conf.item()


        output.append({
            "image": img_path,
            "prediction": pred_name,
            "confidence: ": round(top1_conf,2)
        })

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {os.path.abspath(output_json)}")