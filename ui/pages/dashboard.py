from ui.components.cards import card_start, card_end
from ui.components.metrics import render_metric_row
from ui.components.charts import density_distribution_chart
from ui.helpers import get_analytics


def render(st) -> None:
    st.title("Dashboard")
    st.caption("Live overview of model status and session performance")

    analytics = get_analytics()
    summary = analytics.get_session_summary()

    render_metric_row(st, [
        ("Model Status", "Loaded" if st.session_state.model_loaded else "Not loaded"),
        ("Active Model", st.session_state.weights_path.split("/")[-1]),
        ("Confidence Threshold", f'{st.session_state.get("conf_threshold", 0.5):.2f}'),
        ("Frames Processed", str(summary.get("total_frames", 0))),
    ])

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        card_start(st, "Session Summary")
        if summary:
            st.write(f"**Avg vehicles:** {summary.get('avg_vehicles', 0)}")
            st.write(f"**Peak vehicles:** {summary.get('max_vehicles', 0)}")
            st.write(f"**Std deviation:** {summary.get('std_vehicles', 0)}")
        else:
            st.info("No session data yet. Run detection on the Image or Video pages.")
        card_end(st)

    with col2:
        card_start(st, "Density Distribution")
        density_distribution_chart(st, summary.get("density_distribution", {}))
        card_end(st)

    if st.session_state.last_result:
        card_start(st, "Latest Inference")
        r = st.session_state.last_result
        st.json(r, expanded=False)
        card_end(st)
