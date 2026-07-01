from ui.components.metrics import render_metric_row


def session_summary_cards(st, summary: dict) -> None:
    if not summary:
        st.info("No analytics yet — process an image or video first.")
        return
    render_metric_row(st, [
        ("Total Frames", str(summary.get("total_frames", 0))),
        ("Avg Vehicles", str(summary.get("avg_vehicles", 0))),
        ("Max Vehicles", str(summary.get("max_vehicles", 0))),
        ("Std Dev", str(summary.get("std_vehicles", 0))),
    ])
