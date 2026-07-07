"""
Adversarial Engine Module.
Simulates space-specific conditions to test gesture recognition robustness.
"""

import cv2
import numpy as np


class AdversarialEngine:
    def __init__(self):
        self.lighting_intensity = 0.0
        self.noise_intensity = 0.0
        self.latency_ms = 0
        self.glove_blur = 0
        self.microgravity_jitter = 0.0
        self.frame_buffer = []

    def apply_lighting(self, frame, intensity=None):
        """Simulate lighting changes (bright/dark)."""
        if intensity is None:
            intensity = self.lighting_intensity
        if intensity == 0:
            return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + intensity), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def apply_noise(self, frame, intensity=None):
        """Simulate noisy camera sensor."""
        if intensity is None:
            intensity = self.noise_intensity
        if intensity == 0:
            return frame
        result = frame.copy().astype(np.float32)
        noise = np.random.normal(0, intensity * 80, result.shape)
        result = np.clip(result + noise, 0, 255).astype(np.uint8)
        return result

    def apply_latency(self, frame, latency_ms=None):
        """Simulate processing delay."""
        if latency_ms is None:
            latency_ms = self.latency_ms
        if latency_ms == 0:
            return frame
        frames_behind = int(latency_ms / 33)
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > frames_behind + 5:
            self.frame_buffer.pop(0)
        if len(self.frame_buffer) > frames_behind:
            return self.frame_buffer[-(frames_behind + 1)]
        return frame

    def apply_glove_effect(self, frame, blur_radius=None):
        """Simulate gloved hand (blur)."""
        if blur_radius is None:
            blur_radius = self.glove_blur
        if blur_radius == 0:
            return frame
        k = blur_radius * 2 + 1
        result = cv2.GaussianBlur(frame, (k, k), 0)
        alpha = max(0.5, 1.0 - blur_radius * 0.03)
        return cv2.convertScaleAbs(result, alpha=alpha, beta=10)

    def apply_microgravity(self, landmarks_array, jitter=None):
        """Simulate floating hand instability (applied to landmarks)."""
        if landmarks_array is None:
            return None
        if jitter is None:
            jitter = self.microgravity_jitter
        if jitter == 0:
            return landmarks_array
        noise = np.random.normal(0, jitter, landmarks_array.shape)
        result = landmarks_array + noise
        result[:, 0] = np.clip(result[:, 0], 0, 1)
        result[:, 1] = np.clip(result[:, 1], 0, 1)
        return result

    def apply_all_to_frame(self, frame):
        """Apply all image-based conditions."""
        frame = self.apply_lighting(frame)
        frame = self.apply_noise(frame)
        frame = self.apply_latency(frame)
        frame = self.apply_glove_effect(frame)
        return frame

    def apply_all_to_landmarks(self, landmarks_array):
        """Apply all landmark-based conditions."""
        return self.apply_microgravity(landmarks_array)

    def reset_all(self):
        """Reset all conditions to zero."""
        self.lighting_intensity = 0.0
        self.noise_intensity = 0.0
        self.latency_ms = 0
        self.glove_blur = 0
        self.microgravity_jitter = 0.0
        self.frame_buffer = []