from ui.theme import DENSITY_COLORS


def card_start(st, title: str) -> None:
    st.markdown(f'<div class="tv-card"><h4>{title}</h4>', unsafe_allow_html=True)


def card_end(st) -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def density_card(st, density: dict) -> None:
    """density: output of DensityClassifier.classify(...)"""
    label = density.get("density", "Unknown")
    color = density.get("color") or DENSITY_COLORS.get(label, "#8b98ab")
    count = density.get("vehicle_count", "-")
    rng = density.get("threshold_range", "-")
    st.markdown(
        f"""
        <div class="tv-card">
            <h4>Traffic Density</h4>
            <div style="display:flex;align-items:center;justify-content:space-between;">
                <span class="tv-chip" style="background:{color}22;color:{color};font-size:16px;padding:8px 18px;">
                    ● {label}
                </span>
                <div style="text-align:right;">
                    <div style="font-size:26px;font-weight:700;">{count}</div>
                    <div style="color:#8b98ab;font-size:12px;">vehicles · range {rng}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def congestion_card(st, score: float) -> None:
    score = round(float(score), 1)
    color = "#10b981" if score < 40 else "#f59e0b" if score < 70 else "#ef4444"
    st.markdown(
        f"""
        <div class="tv-card">
            <h4>Congestion Index</h4>
            <div style="font-size:30px;font-weight:700;color:{color};">{score}</div>
            <div style="background:#232d3f;border-radius:6px;height:8px;margin-top:8px;">
                <div style="background:{color};width:{min(score,100)}%;height:8px;border-radius:6px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_chip(st, label: str, ok: bool) -> str:
    from ui.theme import DANGER, SUCCESS
    color = SUCCESS if ok else DANGER
    return f'<span class="tv-chip" style="background:{color}22;color:{color};">● {label}</span>'
