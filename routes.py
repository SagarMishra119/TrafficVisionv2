"""
TrafficVision AI — Streamlit Dashboard
Run: streamlit run streamlit_app.py
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficVision AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #07090f; }
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e293b; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] {
        background: #111820; border: 1px solid #1e293b;
        border-radius: 12px; padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] { font-size: 11px !important; color: #475569 !important; }
    [data-testid="stMetricValue"] { color: #06b6d4 !important; }
    .density-low  { background:#052e1a; color:#10b981; border:1px solid #14532d; padding:4px 14px; border-radius:99px; font-size:13px; font-weight:700; }
    .density-med  { background:#2d1a00; color:#f59e0b; border:1px solid #92400e; padding:4px 14px; border-radius:99px; font-size:13px; font-weight:700; }
    .density-high { background:#2d0a0a; color:#ef4444; border:1px solid #7f1d1d; padding:4px 14px; border-radius:99px; font-size:13px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_detector(weights_path: str):
    from src.detection.detector import VehicleDetector
    return VehicleDetector(weights_path)

@st.cache_resource
def get_counter():
    from src.counting.counter import VehicleCounter
    return VehicleCounter()

@st.cache_resource
def get_classifier():
    from src.density.classifier import DensityClassifier
    return DensityClassifier()

@st.cache_resource
def get_analytics():
    from src.analytics.analytics import TrafficAnalytics
    return TrafficAnalytics()


# ── Helpers ────────────────────────────────────────────────────────────────────
VEHICLE_COLORS = {
    "car": (6, 182, 212), "bus": (245, 158, 11),
    "truck": (139, 92, 246), "motorcycle": (16, 185, 129), "bicycle": (249, 115, 22),
}

def density_badge(level: str, label: str) -> str:
    cls = {"low": "density-low", "medium": "density-med", "high": "density-high"}.get(level, "density-low")
    return f'<span class="{cls}">{label}</span>'

def draw_boxes(image: np.ndarray, detections: list) -> np.ndarray:
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = VEHICLE_COLORS.get(det["vehicle_type"], (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det['vehicle_type']} {det['confidence']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img

def counts_df(counts: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"Class": "🚗 Cars",        "Count": counts["cars"]},
        {"Class": "🚌 Buses",       "Count": counts["buses"]},
        {"Class": "🚛 Trucks",      "Count": counts["trucks"]},
        {"Class": "🏍 Motorcycles", "Count": counts["motorcycles"]},
        {"Class": "🚲 Bicycles",    "Count": counts["bicycles"]},
    ])


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚦 TrafficVision AI")
    st.caption("YOLOv11 · Vehicle Detection & Density Analysis")
    st.divider()

    st.markdown("**Model**")
    weights_input = st.text_input("Weights path", value="models/weights/yolo11n_traffic_best.pt")
    load_btn = st.button("Load Model", type="primary", use_container_width=True)

    if "detector" not in st.session_state:
        st.session_state.detector = None
        st.session_state.model_loaded = False

    if load_btn:
        if Path(weights_input).exists():
            with st.spinner("Loading…"):
                st.session_state.detector = load_detector(weights_input)
                st.session_state.model_loaded = True
            st.success("Model ready!")
        else:
            st.error(f"Not found: {weights_input}")

    status_color = "#10b981" if st.session_state.model_loaded else "#ef4444"
    status_text  = "Model loaded" if st.session_state.model_loaded else "No model — click Load"
    st.markdown(f'<span style="color:{status_color};font-size:12px;">● {status_text}</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Inference**")
    conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)

    st.divider()
    st.markdown("**Density Thresholds**")
    low_max = st.number_input("Low max",    value=10, min_value=1)
    med_max = st.number_input("Medium max", value=25, min_value=2)
    if st.button("Apply Thresholds", use_container_width=True):
        if low_max >= med_max:
            st.error("Low max must be less than Medium max")
        else:
            get_classifier().update_thresholds(int(low_max), int(med_max))
            st.success("Updated!")

    st.divider()
    st.caption("v4.2 · yolo11n / yolo11s")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_dash, tab_image, tab_video, tab_analytics = st.tabs([
    "📊 Dashboard", "🔍 Image Detection", "🎬 Video Analytics", "📈 Session Analytics"
])


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
with tab_dash:
    st.markdown("### Traffic Density Detection & Vehicle Analytics")
    st.caption("YOLOv11 pipeline trained on 80 images · 5 vehicle classes")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Images", "80",  "Train 50 · Val 15 · Test 15")
    c2.metric("Annotations",    "157", "5 classes")
    c3.metric("Active Model",   "yolo11n", "ONNX + PT")
    c4.metric("Model Status",   "Loaded" if st.session_state.model_loaded else "Not loaded")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("**Dataset Class Distribution**")
        dist = pd.DataFrame({
            "Class": ["Car","Motorcycle","Truck","Bus","Bicycle"],
            "Train": [37, 32, 28, 33, 27],
            "Valid": [7,  8,  11, 2,  7],
            "Test":  [16, 12, 11, 10, 8],
        }).set_index("Class")
        st.bar_chart(dist, color=["#06b6d4","#10b981","#8b5cf6"])

    with right:
        st.markdown("**Last Inference Result**")
        with st.container(border=True):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Trucks", 4)
            r2.metric("Buses",  1)
            r3.metric("Total",  5)
            r4.metric("Density","Low")
            st.caption("WhatsApp Image · 2026-06-15 · Congestion index: 21.0")
        st.info("Go to **Image Detection** tab to run a new detection.", icon="🔍")

    st.divider()
    st.markdown("**Pipeline Status**")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.success("✅ Dataset\nValidated")
    p2.success("✅ Trained\nyolo11n + s")
    p3.success("✅ Exported\nONNX")
    p4.success("✅ Inferred\nImage mode")
    p5.info("⚡ Streamlit\nRunning")


# ── IMAGE DETECTION ───────────────────────────────────────────────────────────
with tab_image:
    st.markdown("### Image Detection")

    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp","bmp"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        col_img, col_res = st.columns([2, 1])
        col_img.image(image_rgb, caption="Uploaded image", use_container_width=True)

        with col_res:
            weighted = st.checkbox("Weighted density", help="bus×2.5, truck×2, car×1, moto×0.5, bike×0.3")
            run = st.button("▶ Run Detection", type="primary", use_container_width=True,
                            disabled=not st.session_state.model_loaded)
            if not st.session_state.model_loaded:
                st.warning("Load a model first.")

        if run:
            detector   = st.session_state.detector
            counter    = get_counter()
            classifier = get_classifier()
            analytics  = get_analytics()
            detector.inference_cfg["confidence"] = conf_threshold

            with st.spinner("Detecting…"):
                detections = detector.detect_image(image_bgr)

            counts     = counter.count(detections)
            density    = (classifier.classify_with_weights(counts) if weighted
                          else classifier.classify(counts["total_vehicles"]))
            congestion = analytics.compute_congestion_index(counts)
            analytics.log_frame(0, detections, counts, density)

            annotated = draw_boxes(image_rgb, detections)
            col_img.image(annotated, caption="Detection result", use_container_width=True)

            with col_res:
                st.markdown("**Results**")
                m1, m2 = st.columns(2)
                m1.metric("Vehicles",   counts["total_vehicles"])
                m2.metric("Congestion", f"{congestion:.1f}")

                level = density.get("density_level", "low")
                label = density.get("density", "Low")
                st.markdown(density_badge(level, f"Density: {label}"), unsafe_allow_html=True)
                st.write("")

                if detections:
                    avg_conf = sum(d["confidence"] for d in detections) / len(detections)
                    st.metric("Avg confidence", f"{avg_conf*100:.1f}%")

                st.markdown("**By class**")
                df = counts_df(counts)
                st.dataframe(df[df["Count"] > 0], hide_index=True, use_container_width=True)

                _, buf = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                st.download_button("⬇ Download annotated", data=buf.tobytes(),
                                   file_name="detected.jpg", mime="image/jpeg",
                                   use_container_width=True)
    else:
        st.info("Upload an image to start detection.", icon="📸")


# ── VIDEO ANALYTICS ───────────────────────────────────────────────────────────
with tab_video:
    st.markdown("### Video Analytics")

    video_file = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"])
    max_frames = st.slider("Max frames", 10, 1000, 200, 10)

    if video_file and st.button("▶ Analyze Video", type="primary",
                                 disabled=not st.session_state.model_loaded):
        detector   = st.session_state.detector
        counter    = get_counter()
        classifier = get_classifier()
        analytics  = get_analytics()
        analytics.clear_session()
        detector.inference_cfg["confidence"] = conf_threshold

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        progress = st.progress(0, text="Processing…")
        preview  = st.empty()

        try:
            cap   = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
            cap.release()

            for fid, detections, annotated in detector.detect_video(tmp_path):
                if fid >= max_frames:
                    break
                counts  = counter.count(detections)
                density = classifier.classify(counts["total_vehicles"])
                analytics.log_frame(fid, detections, counts, density)
                progress.progress(min((fid + 1) / min(max_frames, total), 1.0),
                                  text=f"Frame {fid+1} · {counts['total_vehicles']} vehicles")
                if fid % 15 == 0:
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                  caption=f"Frame {fid}", use_container_width=True)
        except Exception as e:
            st.error(f"Video error: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            progress.empty()

        summary = analytics.get_session_summary()
        if summary:
            st.success(f"Done! {summary['total_frames']} frames processed.")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Frames",        summary["total_frames"])
            s2.metric("Avg vehicles",  summary["avg_vehicles"])
            s3.metric("Peak vehicles", summary["max_vehicles"])
            s4.metric("Peak frame",    summary["peak_frame"])

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Density distribution**")
                dist = summary.get("density_distribution", {})
                if dist:
                    st.bar_chart(pd.DataFrame.from_dict(dist, orient="index", columns=["Frames"]))
            with col_b:
                st.markdown("**Avg by type**")
                avg_t = summary.get("avg_by_type", {})
                if avg_t:
                    st.bar_chart(pd.DataFrame.from_dict(avg_t, orient="index", columns=["Avg"]),
                                 color="#06b6d4")

            df_log = pd.DataFrame(analytics._session_log)
            st.download_button("⬇ Download CSV", data=df_log.to_csv(index=False).encode(),
                               file_name="video_analytics.csv", mime="text/csv")

    elif not video_file:
        st.info("Upload a video to start analysis.", icon="🎬")
    elif not st.session_state.model_loaded:
        st.warning("Load a model first.", icon="⚠️")


# ── SESSION ANALYTICS ─────────────────────────────────────────────────────────
with tab_analytics:
    st.markdown("### Session Analytics")
    analytics = get_analytics()

    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with btn2:
        if st.button("🗑 Clear session", use_container_width=True):
            analytics.clear_session()
            st.success("Cleared.")
            st.rerun()

    summary = analytics.get_session_summary()
    if not summary:
        st.info("No session data yet. Run some detections first.", icon="📊")
    else:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total frames",  summary["total_frames"])
        a2.metric("Avg vehicles",  summary["avg_vehicles"])
        a3.metric("Max vehicles",  summary["max_vehicles"])
        a4.metric("Std deviation", summary["std_vehicles"])

        st.divider()
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Density distribution**")
            dist = summary.get("density_distribution", {})
            if dist:
                st.bar_chart(pd.DataFrame.from_dict(dist, orient="index", columns=["Frames"]))
        with col_r:
            st.markdown("**Avg vehicles by type**")
            avg_t = summary.get("avg_by_type", {})
            if avg_t:
                st.bar_chart(pd.DataFrame.from_dict(avg_t, orient="index", columns=["Avg"]),
                             color="#06b6d4")

        st.markdown("**Session log**")
        df_log = pd.DataFrame(analytics._session_log)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download CSV", data=df_log.to_csv(index=False).encode(),
                           file_name="session.csv", mime="text/csv")
