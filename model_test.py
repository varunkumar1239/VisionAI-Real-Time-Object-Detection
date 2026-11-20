print("TEST 1: Python started", flush=True)

import torch
print("TEST 2: PyTorch imported", flush=True)

from ultralytics import YOLO
print("TEST 3: YOLO imported", flush=True)

model = YOLO("yolo26n.pt")
print("TEST 4: Model loaded successfully!", flush=True)