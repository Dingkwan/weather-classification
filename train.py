# train.py
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

# Load a pretrained YOLO11 classification model (large version)
model = YOLO("yolo11l-cls.pt")

# Train the model on the custom weather dataset
model.train(
    data="./weather_split",   # Path to dataset (train/val folders must exist inside)
    epochs=50,
    imgsz=320,
    batch=32,
    device=device
)

# Evaluate the trained model on the test set
print("Running final evaluation on test set...")
metrics = model.val(
    data="./weather_split",    # Path to dataset (test folder must exist inside)
    split="test",
    device=device,
    name="test_result"
)
print(metrics)

# After training, the best model weights will be saved at:
# runs/classify/train/weights/best.pt

# And the testing result details, such as confusion matrix, will be saved at:
# runs/classify/test_result