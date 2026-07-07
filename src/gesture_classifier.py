import numpy as np 
import os
import pickle
from collections import deque

class GestureClassifier:
    def __init__(self, model_path=None, encoder_path=None, buffer_size=15, movement_threshold=0.05):
        self.buffer_size=buffer_size
        self.movement_threshold=movement_threshold
        self.position_buffer=deque(maxlen=buffer_size)
        self.last_gesture="background"
        self.gesture_cooldown=0

        self.model=None
        self.label_encoder=None
        self.use_model=False

        if model_path and os.path.exists(model_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                print(f"[Classifier] Model loaded: {model_path}")
                self.use_model=True
            except Exception as e:
                print(f"[Classifier] Failed to load model: {e}")
        
        if encoder_path and os.path.exists(encoder_path):
            try:
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            except ModuleNotFoundError as e:
                # The pickled encoder depends on a module (e.g. scikit-learn) that's not installed.
                print(f"[Classifier] Could not load label encoder due to missing dependency: {e}. Continuing without encoder.")
                self.label_encoder = None
            except Exception as e:
                print(f"[Classifier] Failed to load label encoder: {e}")
                self.label_encoder = None

    def update(self, landmarks_array): #Add new frame's landmarks to buffer
        if landmarks_array is None:
            self.position_buffer.clear()
            return

        wrist = landmarks_array[0]
        middle_tip = landmarks_array[12]

        self.position_buffer.append({
            "wrist": wrist.copy(),
            "middle_tip": middle_tip.copy(),
        })
    
    def classify(self, landmarks_array=None): # Classify the gesture. Uses model if available, else rule-based.
        if self.use_model and landmarks_array is not None:
            return self._classify_with_model(landmarks_array)
        else:
            return self._classify_rule_based()

    def _classify_with_model(self, landmarks_array): #Classify using the trained CNN+GRU model.
        # Flatten (21,3) to (63,) and add batch dim -> (1,63)
        flat = landmarks_array.flatten().reshape(1, -1)

        # Get prediction probabilities
        prediction = self.model.predict(flat, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]

        # Only accept if confidence > 60%
        if confidence < 0.6:
            return "background"

        # Convert number back to label name
        if self.label_encoder:
            gesture = self.label_encoder.inverse_transform([predicted_class])[0]
        else:
            gesture = str(predicted_class)

        return gesture

    def _classify_rule_based(self): #Classify using hand movement tracking (fallback).
        if len(self.position_buffer) < self.buffer_size // 2:
            return "background"

        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return self.last_gesture

        start_wrist = self.position_buffer[0]["wrist"]
        end_wrist = self.position_buffer[-1]["wrist"]
        start_tip = self.position_buffer[0]["middle_tip"]
        end_tip = self.position_buffer[-1]["middle_tip"]

        dx = end_wrist[0] - start_wrist[0]
        dy = end_wrist[1] - start_wrist[1]
        dz = end_wrist[2] - start_wrist[2]

        tilt_change = (end_tip[1] - end_wrist[1]) - (start_tip[1] - start_wrist[1])
        rotation_change = (end_tip[0] - end_wrist[0]) - (start_tip[0] - start_wrist[0])

        movement = np.sqrt(dx**2 + dy**2)

        if movement < self.movement_threshold and abs(dz) < self.movement_threshold:
            if abs(tilt_change) < 0.03 and abs(rotation_change) < 0.03:
                return "background"

        gesture = self._determine_gesture(dx, dy, dz, tilt_change, rotation_change)

        if gesture != "background":
            self.last_gesture = gesture
            self.gesture_cooldown = 10
            self.position_buffer.clear()

        return gesture

    def _determine_gesture(self, dx, dy, dz, tilt_change, rotation_change):
        """Map movement direction to gesture label."""
        abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)
        abs_tilt, abs_rot = abs(tilt_change), abs(rotation_change)
        max_mov = max(abs_dx, abs_dy, abs_dz, abs_tilt, abs_rot)

        if max_mov == abs_tilt and abs_tilt > 0.03:
            return "pitchup" if tilt_change < 0 else "pitchdown"
        if max_mov == abs_rot and abs_rot > 0.03:
            return "yawleft" if rotation_change < 0 else "yawright"
        if max_mov == abs_dz and abs_dz > self.movement_threshold:
            return "forward" if dz < 0 else "backward"
        if abs_dx > abs_dy:
            if dx > self.movement_threshold:
                return "right"
            elif dx < -self.movement_threshold:
                return "left"
        if abs_dy > abs_dx:
            if dy < -self.movement_threshold:
                return "up"
            elif dy > self.movement_threshold:
                return "down"
        return "background"

    def reset(self):
        """Clear tracking buffer and reset state."""
        self.position_buffer.clear()
        self.last_gesture = "background"
        self.gesture_cooldown = 0