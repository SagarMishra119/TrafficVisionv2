from pathlib import Path

from ui.helpers import load_detector
from ui.theme import DANGER, SUCCESS


def render_sidebar(st) -> str:
    """Renders sidebar; returns the selected page name."""
    with st.sidebar:
        st.markdown("### 🚦 TrafficVision AI")
        st.caption("Traffic density & congestion analytics")
        st.divider()

        page = st.radio(
            "Navigate",
            ["Dashboard", "Image Detection", "Video Analysis", "Session Analytics"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Model**")
        weights_input = st.text_input("Weights path", value=st.session_state.weights_path)
        st.session_state.weights_path = weights_input

        if st.button("Load model", use_container_width=True):
            if Path(weights_input).exists():
                try:
                    st.session_state.detector = load_detector(weights_input)
                    st.session_state.model_loaded = True
                    st.success("Model loaded")
                except Exception as e:
                    st.session_state.model_loaded = False
                    st.error(f"Failed to load: {e}")
            else:
                st.session_state.model_loaded = False
                st.error(f"Not found: {weights_input}")

        color = SUCCESS if st.session_state.model_loaded else DANGER
        text = "Model loaded" if st.session_state.model_loaded else "No model loaded"
        st.markdown(
            f'<span class="tv-chip" style="background:{color}22;color:{color};">'
            f'● {text}</span>',
            unsafe_allow_html=True,
        )

        with st.expander("Advanced settings"):
            st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05, key="conf_threshold")
            st.checkbox("Use weighted density classification", value=False, key="use_weighted")

        st.divider()
        if st.button("Clear session", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_result = None
            get_analytics_safe(st).clear_session()
            st.toast("Session cleared")

    return page


def get_analytics_safe(st):
    from ui.helpers import get_analytics
    return get_analytics()
