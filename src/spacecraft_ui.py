import streamlit as st
class SpacecraftUI:
    def __init__(self):
        # Use session_state so values persist across Streamlit reruns
        if "ship_position" not in st.session_state:
            st.session_state.ship_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        if "ship_orientation" not in st.session_state:
            st.session_state.ship_orientation = {"pitch": 0.0, "yaw": 0.0}
        if "command_log" not in st.session_state:
            st.session_state.command_log = []
    def execute_command(self, gesture, command_text):
        """Update spacecraft state based on gesture."""
        pos = st.session_state.ship_position
        ori = st.session_state.ship_orientation
        step = 1.0       # Translation step size
        rot_step = 5.0   # Rotation step size (degrees)
        # Translation gestures
        if gesture == "up":
            pos["y"] += step
        elif gesture == "down":
            pos["y"] -= step
        elif gesture == "left":
            pos["x"] -= step
        elif gesture == "right":
            pos["x"] += step
        elif gesture == "forward":
            pos["z"] += step
        elif gesture == "backward":
            pos["z"] -= step
        # Rotation gestures
        elif gesture == "pitchup":
            ori["pitch"] += rot_step
        elif gesture == "pitchdown":
            ori["pitch"] -= rot_step
        elif gesture == "yawleft":
            ori["yaw"] -= rot_step
        elif gesture == "yawright":
            ori["yaw"] += rot_step
        # Log command (keep last 20)
        if gesture != "background":
            st.session_state.command_log.append(command_text)
            if len(st.session_state.command_log) > 20:
                st.session_state.command_log.pop(0)
    def render_navigation_panel(self):
        """Render position metrics."""
        st.subheader("🧭 Navigation")
        pos = st.session_state.ship_position
        col1, col2, col3 = st.columns(3)
        col1.metric("X", f"{pos['x']:.1f}")
        col2.metric("Y", f"{pos['y']:.1f}")
        col3.metric("Z", f"{pos['z']:.1f}")
    def render_attitude_panel(self):
        """Render rotation angles."""
        st.subheader("🔄 Attitude Control")
        ori = st.session_state.ship_orientation
        col1, col2 = st.columns(2)
        col1.metric("Pitch", f"{ori['pitch']:.1f}°")
        col2.metric("Yaw", f"{ori['yaw']:.1f}°")
    def render_status_panel(self, gesture):
        """Render current gesture status."""
        st.subheader("📡 Status")
        if gesture == "background":
            st.info("🔵 IDLE — Awaiting gesture command")
        else:
            st.success(f"🟢 ACTIVE — {gesture.upper()}")
    def render_command_log(self):
        """Render command history."""
        st.subheader("📜 Command Log")
        log = st.session_state.command_log
        if log:
            for cmd in reversed(log[-10:]):
                st.text(f"  > {cmd}")
        else:
            st.caption("  No commands yet...")
    def render_all(self, gesture, command_text):
        """Render complete cockpit UI."""
        self.execute_command(gesture, command_text)
        col_left, col_right = st.columns(2)
        with col_left:
            self.render_navigation_panel()
            self.render_attitude_panel()
        with col_right:
            self.render_status_panel(gesture)
            self.render_command_log()