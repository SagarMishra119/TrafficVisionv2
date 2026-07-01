import pandas as pd


def density_timeline(st, history: list[dict]) -> None:
    if not history:
        st.info("No frames processed yet.")
        return
    df = pd.DataFrame(history)
    st.line_chart(df, x="frame_id" if "frame_id" in df else None, y="total_vehicles")


def vehicle_type_breakdown(st, counts: dict) -> None:
    keys = ["cars", "buses", "trucks", "motorcycles", "bicycles"]
    df = pd.DataFrame({"type": keys, "count": [counts.get(k, 0) for k in keys]})
    st.bar_chart(df, x="type", y="count")


def density_distribution_chart(st, distribution: dict) -> None:
    if not distribution:
        st.info("No density data yet.")
        return
    df = pd.DataFrame({"density": list(distribution.keys()), "frames": list(distribution.values())})
    st.bar_chart(df, x="density", y="frames")


def congestion_trend(st, history: list[dict]) -> None:
    if not history:
        st.info("No congestion data yet.")
        return
    df = pd.DataFrame(history)
    if "congestion_index" in df:
        st.line_chart(df, y="congestion_index")
