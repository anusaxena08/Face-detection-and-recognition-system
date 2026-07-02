#!/usr/bin/env python
"""Test MediaPipe Gesture Recognizer."""

import os

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

# Use the gesture recognizer if available
try:
    # Create base options with NO model (will use bundled)
    base_options = base_options_module.BaseOptions()
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)
    print("✓ GestureRecognizer created successfully")
    recognizer.close()
except Exception as e:
    print(f"GestureRecognizer Error: {e}")
    
    # Try HandLandmarker with base_options
    try:
        base_options = base_options_module.BaseOptions()
        options = vision.HandLandmarkerOptions(base_options=base_options)
        landmarker = vision.HandLandmarker.create_from_options(options)
        print("✓ HandLandmarker created successfully")
        landmarker.close()
    except Exception as e2:
        print(f"HandLandmarker Error: {e2}")
