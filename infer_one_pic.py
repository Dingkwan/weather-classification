# infer_one_pic.py
import torch
from ultralytics import YOLO

# Detect available device: MPS (Apple Silicon), CUDA (NVIDIA), or CPU
if torch.backends.mps.is_available():
    device = "mps"   # Apple M1/M2 GPU
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"   # Fallback to CPU

print(f"Using device: {device}")

# Specify trained model weights and test image path
weights_path = "./best.pt"   # Change this to your actual model path
image_path = "./weather_split/test/rainy/rain275.jpg"  # Change this to your image

# Load trained model
model = YOLO(weights_path)

# Run inference
results = model(image_path, device=device)

# Print Top-1 prediction result
for r in results:
    top1_idx = r.probs.top1  # Index of predicted class
    top1_conf = r.probs.top1conf.item()
    cls_name = r.names[top1_idx]
    print(f"Predicted: {cls_name} (confidence {top1_conf:.2f})")