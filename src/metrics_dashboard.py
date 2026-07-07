"""
Metrics Dashboard Module.
Tracks and displays gesture recognition performance metrics in real-time.
"""

import streamlit as st
import numpy as np
from collections import deque
import time


class MetricsDashboard:
    def __init__(self):
        if "metrics" not in st.session_state:
            st.session_state.metrics = {
                "gesture_counts": {},
                "confidence_history": deque(maxlen=200),
                "fps_history": deque(maxlen=50),
                "last_frame_time": time.time(),
                "session_start": time.time(),
                "total_frames": 0,
                "gestures_detected": 0,
            }

    def update(self, gesture, confidence=None):
        """Update metrics with new detection result."""
        m = st.session_state.metrics

        # FPS calculation
        now = time.time()
        dt = now - m["last_frame_time"]
        if dt > 0:
            m["fps_history"].append(1.0 / dt)
        m["last_frame_time"] = now
        m["total_frames"] += 1

        # Gesture counting
        if gesture != "background":
            m["gesture_counts"][gesture] = m["gesture_counts"].get(gesture, 0) + 1
            m["gestures_detected"] += 1

        # Confidence tracking
        if confidence is not None:
            m["confidence_history"].append(confidence)

    def render_summary(self):
        """Render 4 summary metric boxes."""
        m = st.session_state.metrics
        st.subheader("📊 Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        avg_fps = np.mean(list(m["fps_history"])) if m["fps_history"] else 0
        col1.metric("FPS", f"{avg_fps:.1f}")
        col2.metric("Frames", m["total_frames"])
        col3.metric("Gestures", m["gestures_detected"])

        avg_conf = np.mean(list(m["confidence_history"])) if m["confidence_history"] else 0
        col4.metric("Avg Confidence", f"{avg_conf:.1f}%")

    def render_gesture_distribution(self):
        """Render bar chart of gesture frequency."""
        m = st.session_state.metrics
        counts = m["gesture_counts"]

        if counts:
            st.subheader("📈 Gesture Distribution")
            sorted_gestures = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            gestures = [g for g, c in sorted_gestures]
            values = [c for g, c in sorted_gestures]
            st.bar_chart(data=dict(zip(gestures, values)))

    def render_confidence_chart(self):
        """Render line chart of confidence over time."""
        m = st.session_state.metrics
        history = list(m["confidence_history"])

        if len(history) > 5:
            st.subheader("📉 Confidence Over Time")
            st.line_chart(history)

    def render_session_info(self):
        """Render session duration."""
        m = st.session_state.metrics
        duration = time.time() - m["session_start"]
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        st.caption(f"Session: {minutes}m {seconds}s")

    def render_all(self, gesture, confidence=None):
        """Update and render all metrics."""
        self.update(gesture, confidence)
        self.render_summary()
        self.render_gesture_distribution()
        self.render_confidence_chart()
        self.render_session_info()

    def reset(self):
        """Reset all metrics."""
        if "metrics" in st.session_state:
            del st.session_state.metrics
        self.__init__()
