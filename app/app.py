from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from stream_simulator import compute_live_metrics, load_results_dataframe, replay_slice, resolve_results_path


def load_benchmark_table() -> pd.DataFrame:
    candidates = [
        ROOT_DIR / "reports/tables/model_benchmark_all.csv",
        ROOT_DIR / "reports/tables/model_benchmark.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("Replay-Based Anomaly Detection Dashboard")

model_name = st.sidebar.selectbox(
    "Model",
    ["Isolation Forest", "Autoencoder", "Hybrid", "PyOD-HBOS", "AE+HBOS Fusion"],
)
batch_size = st.sidebar.slider("Batch size", min_value=25, max_value=500, value=100, step=25)
window_size = st.sidebar.slider("Trend window size", min_value=50, max_value=1000, value=200, step=50)

results_path = resolve_results_path(model_name=model_name, tables_dir=ROOT_DIR / "reports/tables")
if not results_path.exists():
    st.error(
        f"Results file not found: {results_path}. "
        f"Generate it from pipeline before running dashboard."
    )
    st.stop()

df_all = load_results_dataframe(results_path)
if "cursor" not in st.session_state:
    st.session_state.cursor = 0

col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    if st.button("Next batch"):
        st.session_state.cursor = min(st.session_state.cursor + batch_size, len(df_all))
with col_b:
    if st.button("Reset replay"):
        st.session_state.cursor = 0
with col_c:
    st.write(f"Loaded file: `{results_path}`")

df_live, _ = replay_slice(df_all, st.session_state.cursor, 0)
metrics = compute_live_metrics(df_live, window_size=window_size)

k1, k2, k3 = st.columns(3)
k1.metric("Total processed", f"{metrics['total_rows']}")
k2.metric("Anomaly count", f"{metrics['anomaly_count']}")
k3.metric("Anomaly ratio", f"{metrics['anomaly_ratio']:.3f}")

st.subheader("Confusion Matrix")
cm = metrics["confusion_matrix"]
cm_df = pd.DataFrame(cm, index=["True Normal", "True Attack"], columns=["Pred Normal", "Pred Attack"])
st.dataframe(cm_df, use_container_width=True)

st.subheader("Benchmark Table")
bench_df = load_benchmark_table()
if bench_df.empty:
    st.info("Benchmark table not found yet. Run benchmark pipeline first.")
else:
    st.dataframe(bench_df, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Score Distribution")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df_live["anomaly_score"], bins=40)
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Anomaly Trend")
    trend_df = metrics["trend_df"]
    if trend_df.empty:
        st.info("No data processed yet.")
    else:
        st.line_chart(trend_df.set_index("window")["anomaly_ratio"])

st.subheader("Latest Anomalies")
recent_df = metrics["recent_anomalies"]
if recent_df.empty:
    st.info("No anomalies detected in current replay window.")
else:
    cols = [c for c in ["true_label", "pred_label", "anomaly_score", "if_score", "ae_score"] if c in recent_df.columns]
    st.dataframe(recent_df[cols], use_container_width=True)
