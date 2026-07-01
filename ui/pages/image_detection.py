import json
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from ui.components.cards import card_start, card_end, density_card, congestion_card
from ui.components.detector import run_pipeline_on_image
from ui.components.metrics import render_metric_row
from ui.components.tables import detections_table, counts_df
from ui.components.uploader import image_uploader


def render(st) -> None:
    st.title("Image Detection")
    st.caption("Upload → Preview → Detection → Annotated image → Summary → Download")

    if not st.session_state.model_loaded:
        st.warning("Load a model from the sidebar before running detection.")

    uploaded = image_uploader(st)
    if not uploaded:
        return

    pil_img = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    col_a, col_b = st.columns(2)
    with col_a:
        card_start(st, "Preview")
        st.image(pil_img, use_container_width=True)
        card_end(st)

    run = col_b.container()
    with run:
        card_start(st, "Run Detection")
        st.write(f"Image size: {pil_img.width} × {pil_img.height}")
        run_clicked = st.button(
            "Run detection", type="primary", disabled=not st.session_state.model_loaded
        )
        card_end(st)

    if not run_clicked:
        return
    if not st.session_state.model_loaded:
        st.error("No model loaded.")
        return

    with st.spinner("Running inference..."):
        detections, annotated, counts, density, congestion = run_pipeline_on_image(img_bgr)

    st.session_state.last_result = {
        "timestamp": datetime.now().isoformat(),
        "counts": counts,
        "density": density,
        "congestion_index": congestion,
    }
    st.session_state.history.append({
        "frame_id": len(st.session_state.history) + 1,
        "total_vehicles": counts["total_vehicles"],
        "congestion_index": congestion,
        **counts,
    })

    st.divider()
    st.subheader("Results")

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated", use_container_width=True)

    render_metric_row(st, [
        ("Total Vehicles", str(counts["total_vehicles"])),
        ("Cars", str(counts["cars"])),
        ("Trucks", str(counts["trucks"])),
        ("Buses", str(counts["buses"])),
    ])

    col1, col2 = st.columns(2)
    with col1:
        density_card(st, density)
    with col2:
        congestion_card(st, congestion)

    st.subheader("Vehicle Table")
    detections_table(st, detections)

    dl1, dl2 = st.columns(2)
    with dl1:
        ok, buf = cv2.imencode(".jpg", annotated)
        st.download_button("Download annotated image", buf.tobytes(),
                            file_name="annotated.jpg", mime="image/jpeg", use_container_width=True)
    with dl2:
        st.download_button("Download CSV (counts)", counts_df(counts).to_csv(index=False),
                            file_name="counts.csv", mime="text/csv", use_container_width=True)

    with st.expander("Raw JSON output"):
        st.json({
            "detections": detections,
            "counts": counts,
            "density": density,
            "congestion_index": congestion,
        })
