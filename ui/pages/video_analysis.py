import tempfile
from pathlib import Path

import cv2

from ui.components.cards import card_start, card_end, density_card, congestion_card
from ui.components.charts import density_timeline, vehicle_type_breakdown
from ui.components.metrics import render_metric_row
from ui.components.uploader import video_uploader
from ui.helpers import get_counter, get_classifier, get_analytics


def render(st) -> None:
    st.title("Video Analysis")
    st.caption("Frame-by-frame detection with live progress and analytics")

    if not st.session_state.model_loaded:
        st.warning("Load a model from the sidebar before running detection.")

    uploaded = video_uploader(st)
    if not uploaded:
        return

    max_frames = st.number_input("Max frames to process (0 = all)", min_value=0, value=100, step=10)

    if not st.button("Process video", type="primary", disabled=not st.session_state.model_loaded):
        return

    detector = st.session_state.detector
    counter = get_counter()
    classifier = get_classifier()
    analytics = get_analytics()
    analytics.clear_session()

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    progress = st.progress(0.0, text="Starting...")
    preview_slot = st.empty()
    metrics_slot = st.empty()

    frame_records = []
    frame_id = 0
    limit = max_frames if max_frames > 0 else None

    for fid, detections, annotated in detector.detect_video(video_path):
        if limit and frame_id >= limit:
            break
        counts = counter.count(detections)
        density = classifier.classify(counts["total_vehicles"])
        congestion = analytics.compute_congestion_index(counts)
        analytics.log_frame(fid, detections, counts, density)

        frame_records.append({
            "frame_id": fid,
            "total_vehicles": counts["total_vehicles"],
            "congestion_index": congestion,
            "density": density.get("density"),
            **counts,
        })

        if frame_id % 5 == 0:
            preview_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            metrics_slot.container()
            with metrics_slot.container():
                render_metric_row(st, [
                    ("Frame", str(fid)),
                    ("Vehicles", str(counts["total_vehicles"])),
                    ("Density", density.get("density", "-")),
                    ("Congestion", f'{congestion:.1f}'),
                ])
            if limit:
                progress.progress(min(frame_id / limit, 1.0), text=f"Frame {frame_id}/{limit}")

        frame_id += 1

    progress.progress(1.0, text=f"Done — {frame_id} frames processed")
    st.session_state.history = frame_records

    summary = analytics.get_session_summary()
    st.divider()
    st.subheader("Session Summary")
    render_metric_row(st, [
        ("Total Frames", str(summary.get("total_frames", 0))),
        ("Avg Vehicles", str(summary.get("avg_vehicles", 0))),
        ("Peak Vehicles", str(summary.get("max_vehicles", 0))),
        ("Std Dev", str(summary.get("std_vehicles", 0))),
    ])

    col1, col2 = st.columns(2)
    with col1:
        card_start(st, "Vehicle Trend")
        density_timeline(st, frame_records)
        card_end(st)
    with col2:
        card_start(st, "Vehicle Type Breakdown (last frame)")
        if frame_records:
            vehicle_type_breakdown(st, frame_records[-1])
        card_end(st)

    card_start(st, "Congestion Timeline")
    from ui.components.charts import congestion_trend
    congestion_trend(st, frame_records)
    card_end(st)

    csv_path = analytics.export_to_csv("video_analytics.csv")
    with open(csv_path, "rb") as f:
        st.download_button("Download CSV export", f.read(), file_name="video_analytics.csv",
                            mime="text/csv")
