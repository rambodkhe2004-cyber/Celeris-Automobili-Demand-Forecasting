# Celeris Automobili Industrial — Demand Forecasting & Planning Control Tower

An end-to-end Python project for the supplied Celeris case study. It profiles and cleans ERP history, corrects known stockout demand, creates calendar/lifecycle/event/macro/lag features, compares two baselines with a LightGBM model using a time-based holdout, forecasts the supplied test rows recursively, and produces audit outputs for a Streamlit dashboard.

## 1. Project structure

```text
Celeris_Demand_Forecasting/
├── app.py                         # Streamlit dashboard
├── run_pipeline.py                # One-command pipeline entry point
├── requirements.txt
├── README.md
├── data/
│   └── raw/                       # The seven supplied CSV files
├── src/
│   └── pipeline.py                # Cleaning, features, model, forecast, reports
└── outputs/                       # Generated automatically
```

## 2. Setup (Windows / VS Code)

Install Python 3.10 or newer. Open this folder in VS Code, then run these commands in the terminal:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## 3. Run the complete pipeline

Make sure all seven original CSV files are inside `data/raw/`:

- `train.csv`
- `test.csv`
- `macro_indicators.csv`
- `sku_master.csv`
- `events_calendar.csv`
- `inventory_stockouts.csv`
- `sample_submission.csv`

Then run:

```powershell
python run_pipeline.py
```

The pipeline writes its files to `outputs/`. Training uses LightGBM and can take several minutes depending on CPU and available memory. Do not stop the process while it is running.

## 4. Launch the dashboard

After the pipeline finishes:

```powershell
streamlit run app.py
```

Streamlit opens the local dashboard in your browser. The same sidebar filters (family, SKU, region, channel and customer) apply across the views.

## 5. Main outputs

| Output | Purpose |
|---|---|
| `cleaned_history.csv` | Clean monthly history with stockout-corrected `Demand_Target` |
| `forecast.csv` | Forecast for each Date–SKU_ID–Region row in test data |
| `sample_submission_completed.csv` | Submission-format forecast file |
| `backtest_predictions.csv` | Actual vs baseline/model predictions for the holdout |
| `backtest_metrics_by_segment.csv` | WAPE and bias by family, region and horizon month |
| `data_quality_report.csv` | Data issues, affected row counts and actions |
| `anomaly_register.csv` | Extreme demand, negative quantities, imputed values and stockout corrections |
| `feature_importance.csv` | LightGBM feature importance |
| `metrics.json` | Run metadata, forecast totals and overall backtest metrics |

## 6. Methodology and business-rule alignment

1. `Date` is treated as authoritative; Year and Month are rebuilt from it.
2. Repeated `Record_ID` values retain the latest `Load_Timestamp`.
3. Distinct transaction rows are aggregated to one monthly Date–SKU–Region record.
4. Negative quantities are clipped to zero and flagged. Missing rows are not created as zero-demand observations.
5. Where stockout data matches a monthly key, `Requested_Units` replaces shipped units as the unconstrained demand target.
6. SKU master data supplies product family, business unit and lifecycle attributes. Launch/EOL rules are applied to forecasts.
7. Features include calendar seasonality, lagged demand, rolling statistics, macro indicators and their lags, business events and lifecycle timing.
8. Seasonal-naive and recent-average baselines are compared with LightGBM. The model is trained on log1p demand and recursively predicts the future month by month.

## 7. Important modelling notes

- The brief says stockout records are incomplete. Therefore, corrected demand is available only where a stockout key is present; other history remains based on the ERP target.
- Extreme values are flagged, not automatically deleted. They may represent real one-off customer orders.
- The supplied core pipeline produces a point forecast. P10/P50/P90 intervals, a full rolling-origin 19-month backtest, hierarchical reconciliation and advanced intermittent-demand models are extension tasks, not claimed as completed here.
- Review `anomaly_register.csv` and `data_quality_report.csv` before presenting results. Synthetic case-study data can contain deliberate contradictions that require documented business judgement.
- Backtest scores are estimates on a temporal holdout, not a guarantee of future accuracy. For a rigorous production evaluation, add recursive rolling-origin validation so validation months do not use actual demand from earlier months inside the same forecast horizon.

## 8. Re-run

Delete or move old files in `outputs/` if you want a clean run, then execute `python run_pipeline.py` again. Raw input files are never modified.
