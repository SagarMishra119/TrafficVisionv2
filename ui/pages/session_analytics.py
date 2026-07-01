from ui.components.analytics import session_summary_cards
from ui.components.cards import card_start, card_end
from ui.components.charts import density_distribution_chart, congestion_trend, density_timeline
from ui.components.tables import history_table
from ui.helpers import get_analytics


def render(st) -> None:
    st.title("Session Analytics")
    st.caption("Interactive charts and exportable session-level analytics")

    analytics = get_analytics()
    summary = analytics.get_session_summary()

    session_summary_cards(st, summary)

    col1, col2 = st.columns(2)
    with col1:
        card_start(st, "Density Distribution")
        density_distribution_chart(st, summary.get("density_distribution", {}))
        card_end(st)
    with col2:
        card_start(st, "Vehicle Count Trend")
        density_timeline(st, st.session_state.history)
        card_end(st)

    card_start(st, "Congestion Trend")
    congestion_trend(st, st.session_state.history)
    card_end(st)

    st.subheader("Full Session History")
    history_table(st, st.session_state.history)

    col_a, col_b = st.columns(2)
    with col_a:
        if summary and st.button("Export session analytics (JSON)"):
            path = analytics.export_to_json("session_analytics.json")
            st.success(f"Exported to {path}")
    with col_b:
        if summary and st.button("Export session analytics (CSV)"):
            path = analytics.export_to_csv("session_analytics.csv")
            st.success(f"Exported to {path}")

    if st.button("Clear session", type="secondary"):
        analytics.clear_session()
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()
