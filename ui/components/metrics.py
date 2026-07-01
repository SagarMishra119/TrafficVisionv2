def render_metric_row(st, metrics: list[tuple[str, str]]) -> None:
    """metrics: list of (label, value) tuples, rendered as st.metric in columns."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def render_skeleton_row(st, n: int = 4) -> None:
    cols = st.columns(n)
    for c in cols:
        c.markdown('<div class="tv-skeleton"></div>', unsafe_allow_html=True)
