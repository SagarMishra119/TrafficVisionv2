"""Cached backend loaders + generic helpers. Backend interfaces untouched."""
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_detector(weights_path: str):
    from src.detection.detector import VehicleDetector
    return VehicleDetector(weights_path)


@st.cache_resource(show_spinner=False)
def get_counter():
    from src.counting.counter import VehicleCounter
    return VehicleCounter()


@st.cache_resource(show_spinner=False)
def get_classifier():
    from src.density.classifier import DensityClassifier
    return DensityClassifier()


@st.cache_resource(show_spinner=False)
def get_analytics():
    from src.analytics.analytics import TrafficAnalytics
    return TrafficAnalytics()


def init_session_state(st) -> None:
    defaults = {
        "detector": None,
        "model_loaded": False,
        "weights_path": "models/weights/yolo11n_traffic_best.pt",
        "history": [],       # list of {counts, density, congestion, timestamp}
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
