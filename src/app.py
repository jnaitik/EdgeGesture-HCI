"""
EdgeGesture-HCI — Real-Time Hand Gesture Recognition
Works both locally and on Hugging Face Spaces.
Run locally:  python src/app.py
"""

import gradio as gr
import cv2
import numpy as np
import os
import sys

# ─── Path Setup ───────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, BASE_DIR)

from config import GESTURE_LABELS, GESTURE_TO_COMMAND, MEDIAPIPE_CONFIG
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier

# ─── Load Resources ──────────────────────────────────────────
model_path = os.path.join(BASE_DIR, "models", "gesture_model_robust.h5")
encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
classifier = GestureClassifier(model_path=model_path, encoder_path=encoder_path)
tracker = HandTracker(
    max_hands=MEDIAPIPE_CONFIG["max_num_hands"],
    detection_conf=MEDIAPIPE_CONFIG["min_detection_confidence"],
    tracking_conf=MEDIAPIPE_CONFIG["min_tracking_confidence"],
)

print("[OK] Model and tracker loaded successfully.")


# ─── Frame Processing ────────────────────────────────────────
def process_frame(frame):
    """Process a single webcam frame: detect hand -> classify -> annotate.
    Returns the same frame with landmarks + results drawn on it."""
    if frame is None:
        return frame

    # Convert RGB (Gradio) to BGR (OpenCV)
    img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.flip(img_bgr, 1)  # Mirror

    # Step 1: Detect hand landmarks
    results = tracker.process_frame(img_bgr)
    landmarks = tracker.get_landmark_array(results)

    # Step 2: Classify gesture
    classifier.update(landmarks)
    gesture = classifier.classify(landmarks)
    command = GESTURE_TO_COMMAND.get(gesture, "UNKNOWN")

    # Step 3: Get confidence
    confidence = 0.0
    if classifier.use_model and landmarks is not None:
        flat = landmarks.flatten().reshape(1, -1)
        pred = classifier.model.predict(flat, verbose=0)
        confidence = float(np.max(pred[0])) * 100

    # Step 4: Draw landmarks on frame
    img_bgr = tracker.draw_landmarks(img_bgr, results)

    # Step 5: Clean overlay — large gesture label like reference image
    if gesture != "background":
        label = gesture.upper()
        color = (0, 255, 200)  # Bright cyan-green

        # Large bold gesture text with shadow for visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.8
        thickness = 4
        cv2.putText(img_bgr, label, (22, 62), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img_bgr, label, (20, 60), font, scale, color, thickness, cv2.LINE_AA)

        # Smaller command text below
        cv2.putText(img_bgr, command, (22, 102), font, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img_bgr, command, (20, 100), font, 0.8, (200, 200, 50), 2, cv2.LINE_AA)

        # Confidence
        conf_text = f"{confidence:.0f}%"
        cv2.putText(img_bgr, conf_text, (22, 137), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img_bgr, conf_text, (20, 135), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    # Convert back to RGB for Gradio
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ─── Gesture Command Table ───────────────────────────────────
gesture_table = "| Gesture | Command |\n|---|---|\n"
for g, c in GESTURE_TO_COMMAND.items():
    if g != "background":
        gesture_table += f"| {g} | {c} |\n"


# ─── Gradio Interface ────────────────────────────────────────
demo = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="EdgeGesture-HCI"),
    outputs=gr.Image(label="Live Detection"),
    live=True,
    title="🚀 EdgeGesture-HCI",
    description="**Real-Time Contactless Gesture Control for Space Missions** | ISRO x MANIT Bhopal | CNN-GRU Model | 91.72% Accuracy",
    article=f"""
### 📋 Gesture Commands
{gesture_table}

---

### 🧠 Model Details
- **Architecture:** CNN-GRU Hybrid (3 Conv1D + 1 GRU) | **Accuracy:** 91.72% | **Classes:** 11 | **Size:** ~4MB
- **Augmentation:** 12x (microgravity jitter, scale variation, fingertip occlusion)
- **Designed for ISRO space missions** | Edge-deployable | MediaPipe 21-point hand landmarks

---
*Built at MANIT Bhopal | ISRO-affiliated research project | [GitHub](https://github.com/labhanshgoyal/EdgeGesture-HCI)*
""",
)


# ─── Launch ──────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()