"""TrafficVision AI — Streamlit dashboard entry point.

Only the UI layer was rebuilt. All backend modules (detector, counter,
classifier, analytics) are used unmodified via ui/helpers.py.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from ui.styles import inject_css
from ui.helpers import init_session_state
from ui.components.sidebar import render_sidebar
from ui.pages import dashboard, image_detection, video_analysis, session_analytics

st.set_page_config(
    page_title="TrafficVision AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css(st)
init_session_state(st)

page = render_sidebar(st)

PAGES = {
    "Dashboard": dashboard.render,
    "Image Detection": image_detection.render,
    "Video Analysis": video_analysis.render,
    "Session Analytics": session_analytics.render,
}

PAGES[page](st)
