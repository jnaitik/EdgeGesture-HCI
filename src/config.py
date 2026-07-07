"""
Configuration file for Space Mission HCI System.
Contains all constants, gesture mappings, and UI settings.
"""
#Gestures
GESTURE_LABELS = [
    "up",
    "down",
    "left",
    "right",
    "forward",
    "backward",
    "pitchup",
    "pitchdown",
    "yawleft",
    "yawright",
    "background",
]

#Commands Mapping
GESTURE_TO_COMMAND = {
    "up":        "TRANSLATE UP",
    "down":      "TRANSLATE DOWN",
    "left":      "TRANSLATE LEFT",
    "right":     "TRANSLATE RIGHT",
    "forward":   "THRUST FORWARD",
    "backward":  "THRUST BACKWARD",
    "pitchup":   "PITCH UP (nose up)",
    "pitchdown": "PITCH DOWN (nose down)",
    "yawleft":   "YAW LEFT (turn left)",
    "yawright":  "YAW RIGHT (turn right)",
    "background":"NO COMMAND (idle)",
}

SPACECRAFT_PANELS = [
    "Navigation",
    "Life Support",
    "Communications",
    "Power Systems",
    "Diagnostics",
]

COLORS = {
    "bg_primary":    "#0a0a1a", #black
    "bg_secondary":  "#1a1a2e", #dark blue
    "accent_blue":   "#0f3460", #blue
    "accent_red":    "#e94560", #red
    "accent_green":  "#00d4aa", #teal
    "accent_yellow": "#f5c842", #yellow
    "text_primary":  "#e0e0e0", #white
    "text_secondary":"#a0a0b0", #grey
    "warning":       "#ff6b35", #orange
    "success":       "#00e676", #green
    "danger":        "#ff1744", #red
}

MEDIAPIPE_CONFIG = {
    "max_num_hands":        1,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence":  0.5,
}

ADVERSARIAL_DEFAULTS = {
    "lighting_intensity":    0.0,
    "noise_intensity":       0.0,
    "latency_ms":            0,
    "glove_blur_radius":     0,
    "microgravity_jitter":   0.0,
}