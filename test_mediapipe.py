#!/usr/bin/env python
"""Quick test of MediaPipe Hand Landmarker."""

import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Get the model path
model_path = os.path.join(os.path.dirname(vision.__file__), "bundled_models", "hand_landmarker.task")
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

# Try to create Hand Landmarker with default options
try:
    options = vision.HandLandmarkerOptions()
    landmarker = vision.HandLandmarker.create_from_options(options)
    print("✓ HandLandmarker created successfully")
    landmarker.close()
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
