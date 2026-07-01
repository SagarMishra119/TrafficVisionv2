import pandas as pd


def detections_table(st, detections: list[dict]) -> None:
    if not detections:
        st.info("No vehicles detected.")
        return
    df = pd.DataFrame(detections)
    st.dataframe(df, use_container_width=True, hide_index=True)


def counts_df(counts: dict) -> pd.DataFrame:
    keys = ["total_vehicles", "cars", "buses", "trucks", "motorcycles", "bicycles"]
    return pd.DataFrame({"Metric": keys, "Value": [counts.get(k, 0) for k in keys]})


def history_table(st, history: list[dict]) -> None:
    if not history:
        st.info("No session history yet.")
        return
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
