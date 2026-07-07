"""
Hand Tracker Module.
Wraps MediaPipe Hand Landmarker (Tasks API) to detect and track hand landmarks.
Compatible with MediaPipe 0.10.15+.
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
import urllib.request


class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.5):
        """
        Initialize hand tracker using MediaPipe Tasks API.
        Args:
            max_hands: Maximum number of hands to detect.
            detection_conf: Minimum confidence for detection.
            tracking_conf: Not used in Tasks API (kept for compatibility).
        """
        # Download model file if not present
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "hand_landmarker.task")

        if not os.path.exists(self.model_path):
            print("[HandTracker] Downloading hand landmarker model...")
            MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("[HandTracker] Model downloaded!")

        # Create hand landmarker
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        """Process a video frame and detect hands.
        Args:
            frame: A BGR image (numpy array) from OpenCV.
        Returns:
            results: MediaPipe HandLandmarkerResult.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        return results

    def get_landmark_array(self, results):
        """
        Extract hand landmarks as a NumPy array.
        Args:
            results: The results from process_frame().
        Returns:
            numpy array of shape (21, 3) with [x, y, z] per landmark,
            or None if no hand detected.
        """
        if not results.hand_landmarks:
            return None
        hand_landmarks = results.hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks:
            landmarks.append([lm.x, lm.y, lm.z])
        return np.array(landmarks)

    def draw_landmarks(self, frame, results):
        """
        Draw hand landmarks and connections on the frame.
        Args:
            frame: The BGR image to draw on.
            results: The results from process_frame().
        Returns:
            frame with landmarks drawn on it.
        """
        if not results.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        # Define hand connections (21 landmarks, same as original MediaPipe)
        HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),        # Thumb
            (0,5),(5,6),(6,7),(7,8),        # Index
            (0,9),(9,10),(10,11),(11,12),   # Middle  (fixed: was 5,9)
            (0,13),(13,14),(14,15),(15,16), # Ring    (fixed: was 9,13)
            (0,17),(17,18),(18,19),(19,20), # Pinky   (fixed: was 13,17)
            (5,9),(9,13),(13,17),           # Palm connections
        ]

        for hand_landmarks in results.hand_landmarks:
            # Draw connections (lines between landmarks)
            for start_idx, end_idx in HAND_CONNECTIONS:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 170), 2)

            # Draw landmark points
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 200, 255), -1)
                cv2.circle(frame, (cx, cy), 7, (0, 150, 200), 1)

        return frame

    def get_finger_states(self, landmarks_array):
        """
        Determine which fingers are extended (open) or folded (closed).
        Args:
            landmarks_array: NumPy array of shape (21, 3).
        Returns:
            List of 5 booleans: [thumb, index, middle, ring, pinky].
            True = extended, False = folded.
        """
        if landmarks_array is None:
            return None
        tips = [4, 8, 12, 16, 20]
        pip_joints = [3, 6, 10, 14, 18]
        finger_states = []
        # Thumb: compare x-position (thumb moves sideways)
        if landmarks_array[tips[0]][0] < landmarks_array[pip_joints[0]][0]:
            finger_states.append(True)
        else:
            finger_states.append(False)
        # Other 4 fingers: tip above (lower y) its joint = extended
        for i in range(1, 5):
            if landmarks_array[tips[i]][1] < landmarks_array[pip_joints[i]][1]:
                finger_states.append(True)
            else:
                finger_states.append(False)
        return finger_states

    def release(self):
        """Release MediaPipe resources."""
        self.detector.close()
