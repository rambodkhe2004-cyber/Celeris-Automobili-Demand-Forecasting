"""Streamlit planning control tower. Run: streamlit run app.py"""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
st.set_page_config(page_title="Celeris | Demand Control Tower", page_icon="📊", layout="wide")
st.title("Celeris Automobili Industrial")
st.caption("Demand Forecasting & Planning Control Tower | Historical demand + 19-month forecast")

@st.cache_data(show_spinner="Loading pipeline outputs…")
def load_csv(name):
    path = OUT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_metrics():
    p = OUT / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

metrics = load_metrics()
required = ["cleaned_history.csv", "forecast.csv", "data_quality_report.csv", "anomaly_register.csv", "backtest_predictions.csv"]
missing = [f for f in required if not (OUT / f).exists()]
if missing:
    st.warning("Pipeline outputs are not ready. First run `python run_pipeline.py` from the project folder.")
    st.code("pip install -r requirements.txt\npython run_pipeline.py\nstreamlit run app.py", language="bash")
    st.stop()

history = load_csv("cleaned_history.csv")
forecast = load_csv("forecast.csv")
dq = load_csv("data_quality_report.csv")
anomalies = load_csv("anomaly_register.csv")
backtest = load_csv("backtest_predictions.csv")
importance = load_csv("feature_importance.csv")

history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
forecast["Date"] = pd.to_datetime(forecast["Date"], errors="coerce")
backtest["Date"] = pd.to_datetime(backtest["Date"], errors="coerce")

# Shared filters
st.sidebar.header("Shared filters")
def options(df, col):
    return sorted(df[col].dropna().astype(str).unique().tolist()) if col in df else []
filters = {}
for col, label in [("Product_Family", "Family"), ("SKU_ID", "SKU"), ("Region", "Region"), ("Sales_Channel", "Channel"), ("Customer_Name", "Customer")]:
    available = options(history, col)
    selected = st.sidebar.multiselect(label, available, default=[])
    filters[col] = selected

def apply_filters(df):
    out = df.copy()
    for col, selected in filters.items():
        if selected and col in out:
            out = out[out[col].astype(str).isin(selected)]
    return out

h = apply_filters(history)
f = apply_filters(forecast.merge(history[["SKU_ID", "Region", "Product_Family", "Sales_Channel", "Customer_Name"]].drop_duplicates(["SKU_ID", "Region"]), on=["SKU_ID", "Region"], how="left"))
b = apply_filters(backtest)
a = apply_filters(anomalies)

# Tabs share the same filters.
t1, t2, t3, t4, t5 = st.tabs(["Executive overview", "Data health", "Demand explorer", "Anomaly monitor", "Forecast & accuracy"])

with t1:
    hist_units = h.groupby("Date", as_index=False)["Demand_Target"].sum().rename(columns={"Demand_Target": "Units"})
    hist_units["Series"] = "Actual / corrected demand"
    fut_units = f.groupby("Date", as_index=False)["Units_Sold_Forecast"].sum().rename(columns={"Units_Sold_Forecast": "Units"})
    fut_units["Series"] = "Forecast"
    timeline = pd.concat([hist_units, fut_units], ignore_index=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("19-month forecast units", f"{int(f['Units_Sold_Forecast'].sum()):,}")
    c2.metric("Historical demand units", f"{int(h['Demand_Target'].sum()):,}")
    c3.metric("SKU-region series", f"{f[['SKU_ID','Region']].drop_duplicates().shape[0]:,}")
    st.subheader("Demand history → forecast")
    st.plotly_chart(px.line(timeline, x="Date", y="Units", color="Series", title="Monthly units — forecast starts after Oct 2026"), use_container_width=True)
    c1, c2 = st.columns(2)
    mix = h.groupby("Product_Family", as_index=False)["Demand_Target"].sum().sort_values("Demand_Target", ascending=False)
    c1.plotly_chart(px.bar(mix, x="Product_Family", y="Demand_Target", title="Historical demand by family"), use_container_width=True)
    region_mix = h.groupby("Region", as_index=False)["Demand_Target"].sum().sort_values("Demand_Target", ascending=False)
    c2.plotly_chart(px.pie(region_mix, names="Region", values="Demand_Target", title="Historical demand mix by region"), use_container_width=True)
    if metrics:
        with st.expander("Overall model metrics"):
            st.json(metrics.get("backtest", {}))

with t2:
    st.subheader("Data quality report")
    st.dataframe(dq, use_container_width=True, hide_index=True)
    st.metric("Clean monthly SKU-region rows", f"{len(history):,}")
    st.metric("Quality issues / audit flags", f"{len(anomalies):,}")
    st.download_button("Download data quality report", dq.to_csv(index=False).encode(), "data_quality_report.csv", "text/csv")

with t3:
    st.subheader("Demand explorer")
    if h.empty:
        st.info("No historical rows match the selected filters.")
    else:
        grain = st.selectbox("View trend by", ["SKU_ID", "Region", "Product_Family", "Customer_Name"])
        group_cols = ["Date", grain]
        trend = h.groupby(group_cols, as_index=False)["Demand_Target"].sum()
        st.plotly_chart(px.line(trend, x="Date", y="Demand_Target", color=grain, title=f"Demand trend by {grain}"), use_container_width=True)
        c1, c2 = st.columns(2)
        zero = h.assign(Zero_Month=h["Demand_Target"].eq(0)).groupby("SKU_ID", as_index=False)["Zero_Month"].mean()
        c1.plotly_chart(px.histogram(zero, x="Zero_Month", nbins=20, title="SKU zero-month rate distribution"), use_container_width=True)
        season = h.assign(Month=h["Date"].dt.month).groupby("Month", as_index=False)["Demand_Target"].mean()
        c2.plotly_chart(px.bar(season, x="Month", y="Demand_Target", title="Average demand by calendar month"), use_container_width=True)
        st.dataframe(h.sort_values("Date", ascending=False).head(500), use_container_width=True)

with t4:
    st.subheader("Anomaly register")
    if a.empty:
        st.success("No anomaly records for the selected filters.")
    else:
        counts = a.groupby("Anomaly_Type", as_index=False).size().sort_values("size", ascending=False)
        st.plotly_chart(px.bar(counts, x="Anomaly_Type", y="size", title="Flagged records by anomaly type"), use_container_width=True)
        st.dataframe(a, use_container_width=True, hide_index=True)
    st.download_button("Download anomaly register", a.to_csv(index=False).encode(), "anomaly_register_filtered.csv", "text/csv")

with t5:
    st.subheader("Forecast & backtest accuracy")
    if metrics.get("backtest"):
        bm = metrics["backtest"]
        c1, c2, c3 = st.columns(3)
        c1.metric("LightGBM WAPE", f"{bm.get('lightgbm_wape', float('nan')):.1%}")
        c2.metric("LightGBM bias", f"{bm.get('lightgbm_bias', float('nan')):.1%}")
        c3.metric("Validation window", f"{bm.get('validation_start')} to {bm.get('validation_end')}")
    if not b.empty:
        plot = b.groupby("Date", as_index=False)[["Demand_Target", "Model_Forecast", "Baseline_Seasonal", "Baseline_Recent"]].sum()
        st.plotly_chart(px.line(plot, x="Date", y=["Demand_Target", "Model_Forecast", "Baseline_Seasonal", "Baseline_Recent"], title="Backtest: actual vs model and baselines"), use_container_width=True)
    if not importance.empty:
        st.plotly_chart(px.bar(importance.head(20).sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Top 20 feature importances"), use_container_width=True)
    st.caption("Point forecasts are supplied. Prediction intervals (P10/P90) are not estimated in this Core pipeline.")
    st.dataframe(f, use_container_width=True, hide_index=True)
    st.download_button("Download filtered forecast", f.to_csv(index=False).encode(), "forecast_filtered.csv", "text/csv")

st.sidebar.divider()
st.sidebar.caption("Data source: local pipeline outputs. Refresh by rerunning the pipeline.")
