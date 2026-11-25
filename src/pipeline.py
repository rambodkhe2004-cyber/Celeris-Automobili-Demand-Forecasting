"""End-to-end demand forecasting pipeline for the Celeris case study.

Run from the project root:
    python run_pipeline.py
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["Date", "SKU_ID", "Region"]
MACRO_COLS = [
    "ISM_Manufacturing_PMI",
    "Auto_Production_YoY_Pct",
    "Vehicle_Inventory_Days",
    "Raw_Material_Cost_Index",
    "New_Vehicle_Orders_Index",
    "Industrial_Util_Rate_Pct",
    "Fuel_Price_Index",
    "EV_Adoption_Index",
]
CAT_COLS = [
    "Product_Family", "Business_Unit", "SKU_ID", "Region", "Sales_Channel",
    "Customer_Name", "Size_Class", "Powertrain_Application", "Lifecycle_Status",
    "Replaces_SKU",
]
LAG_COLS = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_18", "lag_24"]
ROLL_COLS = [
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "rolling_median_12",
    "rolling_std_6", "zero_rate_12", "observations_12", "expanding_mean",
]
EVENT_COLS = [
    "event_count", "event_holiday", "event_disruption", "event_supply_constraint",
    "event_price_change", "event_customer", "event_marketing", "event_other",
]


def _parse_one_month(value: Any) -> pd.Timestamp | pd.NaT:
    """Parse noisy month labels into first-of-month timestamps."""
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if not s or s.lower() in {"nan", "nat", "none", "null"}:
        return pd.NaT

    # YYYY-MM, YYYY/MM, or YYYY-MM-DD.
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?", s)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return pd.Timestamp(year=year, month=month, day=1)

    # MM-YYYY or MM/YYYY (e.g. 01-2022, 05/2025).
    m = re.fullmatch(r"(\d{1,2})[-/](\d{4})", s)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return pd.Timestamp(year=year, month=month, day=1)

    # Month names such as Apr 2024, Mar-2025, December 2022.
    try:
        parsed = pd.to_datetime(s, errors="raise")
        return pd.Timestamp(year=parsed.year, month=parsed.month, day=1)
    except Exception:
        return pd.NaT


def parse_month_series(series: pd.Series) -> pd.Series:
    return series.map(_parse_one_month)


def _clean_text(series: pd.Series, default: str = "Unknown") -> pd.Series:
    out = series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return out.fillna(default)


def normalize_region(series: pd.Series) -> pd.Series:
    s = _clean_text(series).str.lower().str.replace(r"[^a-z]", "", regex=True)
    mapping = {
        "northamerica": "North America", "namerica": "North America",
        "europe": "Europe", "eu": "Europe",
        "asiapacific": "Asia Pacific", "apac": "Asia Pacific",
        "india": "India",
        "latinamerica": "Latin America", "latam": "Latin America",
        "middleeastafrica": "Middle East & Africa", "mea": "Middle East & Africa",
    }
    return s.map(mapping).fillna("Unknown")


def normalize_channel(series: pd.Series) -> pd.Series:
    s = _clean_text(series).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    mapping = {
        "directoem": "Direct OEM", "oemdirect": "Direct OEM",
        "tier1distributor": "Tier-1 Distributor", "t1distributor": "Tier-1 Distributor",
        "aftermarket": "Aftermarket", "export": "Export",
    }
    return s.map(mapping).fillna("Unknown")


def normalize_customer(series: pd.Series) -> pd.Series:
    s = _clean_text(series)
    s = s.str.replace(r"(?i)\s+inc\.?$", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    # Canonicalize case while preserving readable labels.
    return s.map(lambda x: x.title() if x != "Unknown" else x)


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.replace("$", "", regex=False)
        .replace({"-": pd.NA, "": pd.NA, "nan": pd.NA, "None": pd.NA}),
        errors="coerce",
    )


def _month_diff(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


def _load_inputs() -> dict[str, pd.DataFrame]:
    required = [
        "train.csv", "test.csv", "macro_indicators.csv", "sku_master.csv",
        "events_calendar.csv", "inventory_stockouts.csv", "sample_submission.csv",
    ]
    missing = [f for f in required if not (RAW / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s) in {RAW}: {missing}")
    return {
        "train": pd.read_csv(RAW / "train.csv", low_memory=False),
        "test": pd.read_csv(RAW / "test.csv", low_memory=False),
        "macro": pd.read_csv(RAW / "macro_indicators.csv", low_memory=False),
        "sku": pd.read_csv(RAW / "sku_master.csv", low_memory=False),
        "events": pd.read_csv(RAW / "events_calendar.csv", low_memory=False),
        "stockouts": pd.read_csv(RAW / "inventory_stockouts.csv", low_memory=False),
        "sample": pd.read_csv(RAW / "sample_submission.csv", low_memory=False),
    }


def _prepare_sku_master(sku: pd.DataFrame) -> pd.DataFrame:
    sku = sku.copy()
    sku["SKU_ID"] = _clean_text(sku["SKU_ID"])
    for c in ["Product_Family", "Business_Unit", "SKU_Description", "Size_Class", "Powertrain_Application", "Lifecycle_Status"]:
        sku[c] = _clean_text(sku[c])
    sku["Replaces_SKU"] = _clean_text(sku["Replaces_SKU"], default="None")
    sku.loc[sku["Replaces_SKU"].eq("Unknown"), "Replaces_SKU"] = "None"
    sku["Launch_Date"] = parse_month_series(sku["Launch_Date"])
    sku["EOL_Date"] = parse_month_series(sku["EOL_Date"])
    return sku.drop_duplicates("SKU_ID", keep="last")


def _prepare_stockouts(stock: pd.DataFrame) -> pd.DataFrame:
    x = stock.copy()
    x["Date"] = parse_month_series(x["Date"])
    x["SKU_ID"] = _clean_text(x["SKU_ID"])
    x["Region"] = normalize_region(x["Region"])
    for c in ["Requested_Units", "Carried_In_Units", "Shipped_Units", "Backordered_Units", "Lost_Units", "Fill_Rate_Pct", "Stockout_Days"]:
        x[c] = numeric_series(x[c])
    x = x.dropna(subset=["Date", "SKU_ID"])
    x = x[x["Region"].ne("Unknown")]
    # The file is expected to have one record per key; sum if a duplicate is present.
    x = x.groupby(KEYS, as_index=False).agg({
        "Requested_Units": "sum", "Carried_In_Units": "sum", "Shipped_Units": "sum",
        "Backordered_Units": "sum", "Lost_Units": "sum", "Fill_Rate_Pct": "mean",
        "Stockout_Days": "max", "Reason": "last",
    })
    return x


def _prepare_macro(macro: pd.DataFrame) -> pd.DataFrame:
    x = macro.copy()
    x["Date"] = parse_month_series(x["Date"])
    x = x.dropna(subset=["Date"]).sort_values("Date")
    for c in MACRO_COLS:
        x[c] = numeric_series(x[c])
    x = x.groupby("Date", as_index=False)[MACRO_COLS].mean()
    full = pd.DataFrame({"Date": pd.date_range(x["Date"].min(), x["Date"].max(), freq="MS")})
    x = full.merge(x, on="Date", how="left").sort_values("Date")
    x[MACRO_COLS] = x[MACRO_COLS].interpolate(limit_direction="both").ffill().bfill()
    for col in MACRO_COLS:
        for lag in [1, 3, 6]:
            x[f"{col}_lag{lag}"] = x[col].shift(lag)
    return x


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    x = events.copy()
    x["Start_Date"] = parse_month_series(x["Start_Date"])
    x["End_Date"] = parse_month_series(x["End_Date"])
    x["Scope_Type"] = _clean_text(x["Scope_Type"]).str.lower()
    x["Scope_Value"] = _clean_text(x["Scope_Value"])
    x["Event_Type"] = _clean_text(x["Event_Type"]).str.lower()
    x = x.dropna(subset=["Start_Date"])
    x["End_Date"] = x["End_Date"].fillna(pd.Timestamp("2035-12-01"))
    x["Start_Date"] = x["Start_Date"].dt.to_period("M").dt.to_timestamp()
    x["End_Date"] = x["End_Date"].dt.to_period("M").dt.to_timestamp()
    return x


def _apply_event_features(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    x = panel.copy()
    for c in EVENT_COLS:
        x[c] = 0.0
    for _, ev in events.iterrows():
        mask = x["Date"].between(ev["Start_Date"], ev["End_Date"])
        scope = str(ev["Scope_Type"]).lower()
        scope_value = str(ev["Scope_Value"]).strip()
        if scope == "global":
            pass
        elif scope == "region":
            val = normalize_region(pd.Series([scope_value])).iloc[0]
            mask &= x["Region"].eq(val)
        elif scope in {"product_family", "product family"}:
            mask &= x["Product_Family"].astype(str).str.casefold().eq(scope_value.casefold())
        elif scope == "channel":
            val = normalize_channel(pd.Series([scope_value])).iloc[0]
            mask &= x["Sales_Channel"].eq(val)
        elif scope == "customer":
            # Customer events can be supplied as "Customer | Region".
            pieces = [p.strip() for p in scope_value.split("|")]
            customer = normalize_customer(pd.Series([pieces[0]])).iloc[0]
            mask &= x["Customer_Name"].eq(customer)
            if len(pieces) > 1:
                region = normalize_region(pd.Series([pieces[1]])).iloc[0]
                mask &= x["Region"].eq(region)
        else:
            continue
        et = str(ev["Event_Type"]).lower()
        if "holiday" in et:
            bucket = "event_holiday"
        elif "disrupt" in et or "logistics" in et or "system" in et:
            bucket = "event_disruption"
        elif "supply" in et:
            bucket = "event_supply_constraint"
        elif "price" in et:
            bucket = "event_price_change"
        elif "customer" in et:
            bucket = "event_customer"
        elif "marketing" in et or "promotion" in et:
            bucket = "event_marketing"
        else:
            bucket = "event_other"
        x.loc[mask, bucket] += 1.0
        x.loc[mask, "event_count"] += 1.0
    return x


def _prepare_train(train: pd.DataFrame, sku: pd.DataFrame, stock: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    x = train.copy()
    metrics: dict[str, int] = {"train_rows_raw": int(len(x))}
    x["_raw_year"] = pd.to_numeric(x["Year"], errors="coerce")
    x["_raw_month"] = pd.to_numeric(x["Month"], errors="coerce")
    x["Date"] = parse_month_series(x["Date"])
    metrics["invalid_date_rows_dropped"] = int(x["Date"].isna().sum())
    x = x.dropna(subset=["Date", "SKU_ID"]).copy()
    x["Date"] = x["Date"].dt.to_period("M").dt.to_timestamp()
    x["SKU_ID"] = _clean_text(x["SKU_ID"])
    x["Region"] = normalize_region(x["Region"])
    x["Sales_Channel"] = normalize_channel(x["Sales_Channel"])
    x["Customer_Name"] = normalize_customer(x["Customer_Name"])
    x["Units_Sold_Raw"] = x["Units_Sold"].astype("string")
    x["Units_Sold_Numeric"] = numeric_series(x["Units_Sold"])
    x["Unit_Price_USD_Raw"] = pd.to_numeric(x["Unit_Price_USD"], errors="coerce")
    x["Revenue_USD_Raw"] = pd.to_numeric(x["Revenue_USD"], errors="coerce")
    x["Load_Timestamp"] = pd.to_datetime(x["Load_Timestamp"], errors="coerce")
    metrics["missing_units_rows"] = int(x["Units_Sold_Numeric"].isna().sum())
    metrics["non_numeric_units_rows"] = int((x["Units_Sold"].notna() & x["Units_Sold_Numeric"].isna()).sum())
    metrics["negative_units_rows"] = int((x["Units_Sold_Numeric"] < 0).sum())
    metrics["year_month_disagreement_rows"] = int(
        ((x["_raw_year"] != x["Date"].dt.year) | (x["_raw_month"] != x["Date"].dt.month)).sum()
    )

    # Business rule: the latest load for a repeated Record_ID is the valid row.
    before = len(x)
    x = x.sort_values("Load_Timestamp").drop_duplicates("Record_ID", keep="last")
    metrics["duplicate_record_id_rows_removed"] = int(before - len(x))

    # SKU master is the authoritative reference for product family / BU / description.
    master_cols = ["SKU_ID", "Product_Family", "Business_Unit", "SKU_Description", "Size_Class",
                   "Powertrain_Application", "Launch_Date", "EOL_Date", "Replaces_SKU", "Lifecycle_Status"]
    x = x.merge(sku[master_cols], on="SKU_ID", how="left", suffixes=("_source", ""))
    for c in ["Product_Family", "Business_Unit", "SKU_Description"]:
        source_c = f"{c}_source"
        if source_c in x.columns:
            x[c] = x[c].fillna(x[source_c])
            x.drop(columns=source_c, inplace=True)
    for c in ["Product_Family", "Business_Unit", "SKU_Description", "Size_Class", "Powertrain_Application", "Lifecycle_Status"]:
        x[c] = _clean_text(x[c])
    x["Replaces_SKU"] = _clean_text(x["Replaces_SKU"], default="None").replace({"Unknown": "None"})

    x["Units_Sold_Corrected"] = x["Units_Sold_Numeric"].clip(lower=0)
    x["Negative_Units_Flag"] = x["Units_Sold_Numeric"].lt(0)

    # Aggregate distinct transaction IDs to the required monthly SKU-region grain.
    before_keys = x.duplicated(KEYS, keep=False).sum()
    x = x.sort_values("Load_Timestamp")
    x["_price_valid"] = x["Unit_Price_USD_Raw"].where(x["Unit_Price_USD_Raw"] > 0)
    x["_price_weight"] = x["Units_Sold_Corrected"].fillna(0).clip(lower=0).where(x["_price_valid"].notna(), 0)
    x["_weighted_price"] = x["_price_valid"].fillna(0) * x["_price_weight"]
    grouped = x.groupby(KEYS, sort=False, dropna=False)
    qty = grouped["Units_Sold_Corrected"].sum(min_count=1).rename("Units_Sold_Clean")
    price_num = grouped["_weighted_price"].sum()
    price_den = grouped["_price_weight"].sum()
    price_mean = grouped["_price_valid"].mean()
    price = (price_num / price_den).where(price_den > 0, price_mean).rename("Unit_Price_USD_Clean")
    latest = x.drop_duplicates(KEYS, keep="last").set_index(KEYS)
    negative_flags = grouped["Negative_Units_Flag"].max().rename("Negative_Units_Flag")
    missing_flags = x.assign(_missing_units=x["Units_Sold_Corrected"].isna()).groupby(KEYS, sort=False)["_missing_units"].any().rename("Missing_Units_Flag")
    flags = pd.concat([negative_flags, missing_flags], axis=1)
    clean = pd.concat([qty, price, latest[["Customer_Name", "Sales_Channel", "Product_Family", "Business_Unit",
                                           "SKU_Description", "Size_Class", "Powertrain_Application", "Lifecycle_Status",
                                           "Replaces_SKU", "Launch_Date", "EOL_Date", "Load_Timestamp"]], flags], axis=1).reset_index()
    metrics["duplicate_key_rows_aggregated"] = int(before_keys)

    # Stockout file provides unconstrained requested demand.
    stock = stock.copy()
    clean = clean.merge(stock[[*KEYS, "Requested_Units", "Shipped_Units", "Backordered_Units", "Lost_Units", "Stockout_Days", "Reason"]],
                        on=KEYS, how="left")
    clean["Stockout_Flag"] = clean["Requested_Units"].notna()
    clean["Demand_Target"] = pd.to_numeric(clean["Units_Sold_Clean"], errors="coerce").astype(float)
    clean.loc[clean["Stockout_Flag"], "Demand_Target"] = pd.to_numeric(clean.loc[clean["Stockout_Flag"], "Requested_Units"], errors="coerce").astype(float)
    metrics["stockout_rows_matched"] = int(clean["Stockout_Flag"].sum())
    metrics["stockout_demand_uplift_units"] = float(
        (clean.loc[clean["Stockout_Flag"], "Demand_Target"] - clean.loc[clean["Stockout_Flag"], "Units_Sold_Clean"].fillna(0)).clip(lower=0).sum()
    )

    # Missing values are not interpreted as zero. Impute only missing values on observed ERP rows.
    missing_mask = clean["Demand_Target"].isna()
    metrics["missing_demand_to_impute"] = int(missing_mask.sum())
    if missing_mask.any():
        clean["_month"] = clean["Date"].dt.month
        seasonal = clean.groupby(["SKU_ID", "Region", "_month"])["Demand_Target"].transform("median")
        series_med = clean.groupby(["SKU_ID", "Region"])["Demand_Target"].transform("median")
        sku_med = clean.groupby("SKU_ID")["Demand_Target"].transform("median")
        family_med = clean.groupby(["Product_Family", "Region"])["Demand_Target"].transform("median")
        fill = seasonal.fillna(series_med).fillna(sku_med).fillna(family_med).fillna(clean["Demand_Target"].median())
        clean.loc[missing_mask, "Demand_Target"] = fill.loc[missing_mask]
        clean["Demand_Imputed_Flag"] = missing_mask
        clean.drop(columns="_month", inplace=True)
    else:
        clean["Demand_Imputed_Flag"] = False

    # Impute invalid / zero prices only for positive-demand records, respecting SKU/year price levels.
    clean["_year"] = clean["Date"].dt.year
    price_bad = clean["Unit_Price_USD_Clean"].isna() | (clean["Unit_Price_USD_Clean"] <= 0)
    metrics["invalid_or_zero_price_rows"] = int(price_bad.sum())
    positive_prices = clean[clean["Unit_Price_USD_Clean"] > 0]
    sku_year_med = positive_prices.groupby(["SKU_ID", "_year"])["Unit_Price_USD_Clean"].median()
    sku_med_price = positive_prices.groupby("SKU_ID")["Unit_Price_USD_Clean"].median()
    global_price = float(positive_prices["Unit_Price_USD_Clean"].median()) if len(positive_prices) else 1.0
    for idx in clean.index[price_bad]:
        row = clean.loc[idx]
        replacement = sku_year_med.get((row["SKU_ID"], row["_year"]), np.nan)
        if pd.isna(replacement):
            replacement = sku_med_price.get(row["SKU_ID"], global_price)
        clean.at[idx, "Unit_Price_USD_Clean"] = replacement
    clean["Revenue_USD_Clean"] = clean["Demand_Target"] * clean["Unit_Price_USD_Clean"]
    clean["Year"] = clean["Date"].dt.year
    clean["Month"] = clean["Date"].dt.month
    clean["Date"] = clean["Date"].dt.strftime("%Y-%m")
    clean.drop(columns=["_year"], inplace=True)

    clean["Anomaly_Source_Flag"] = (
        clean["Negative_Units_Flag"] | clean["Missing_Units_Flag"] |
        clean["Demand_Imputed_Flag"] | clean["Stockout_Flag"]
    )
    return clean, metrics, x


def _prepare_test(test: pd.DataFrame, sku: pd.DataFrame) -> pd.DataFrame:
    x = test.copy()
    x["Date"] = parse_month_series(x["Date"])
    x = x.dropna(subset=["Date", "SKU_ID"]).copy()
    x["Date"] = x["Date"].dt.to_period("M").dt.to_timestamp()
    x["SKU_ID"] = _clean_text(x["SKU_ID"])
    x["Region"] = normalize_region(x["Region"])
    x["Sales_Channel"] = normalize_channel(x["Sales_Channel"])
    x["Customer_Name"] = normalize_customer(x["Customer_Name"])
    master_cols = ["SKU_ID", "Product_Family", "Business_Unit", "SKU_Description", "Size_Class",
                   "Powertrain_Application", "Launch_Date", "EOL_Date", "Replaces_SKU", "Lifecycle_Status"]
    x = x.merge(sku[master_cols], on="SKU_ID", how="left", suffixes=("_source", ""))
    for c in ["Product_Family", "Business_Unit", "SKU_Description"]:
        src = f"{c}_source"
        if src in x.columns:
            x[c] = x[c].fillna(x[src])
            x.drop(columns=src, inplace=True)
    for c in ["Product_Family", "Business_Unit", "SKU_Description", "Size_Class", "Powertrain_Application", "Lifecycle_Status"]:
        x[c] = _clean_text(x[c])
    x["Replaces_SKU"] = _clean_text(x["Replaces_SKU"], default="None").replace({"Unknown": "None"})
    x["Year"] = x["Date"].dt.year
    x["Month"] = x["Date"].dt.month
    if x.duplicated(KEYS).any():
        x = x.drop_duplicates(KEYS, keep="last")
    return x


def _build_panel(clean: pd.DataFrame, test: pd.DataFrame, sku: pd.DataFrame, stock: pd.DataFrame,
                 macro: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = clean.copy()
    hist["Date"] = pd.to_datetime(hist["Date"] + "-01")
    future = test.copy()
    series = pd.concat([hist[KEYS[1:]], future[KEYS[1:]]], ignore_index=True).drop_duplicates()
    series = series.sort_values(KEYS[1:]).reset_index(drop=True)
    series["series_id"] = np.arange(len(series), dtype=np.int32)
    months = pd.date_range(hist["Date"].min(), future["Date"].max(), freq="MS")
    panel = series.assign(_join=1).merge(pd.DataFrame({"Date": months, "_join": 1}), on="_join").drop(columns="_join")

    hist_cols = [*KEYS, "Demand_Target", "Units_Sold_Clean", "Unit_Price_USD_Clean", "Revenue_USD_Clean",
                 "Customer_Name", "Sales_Channel", "Product_Family", "Business_Unit", "SKU_Description",
                 "Size_Class", "Powertrain_Application", "Lifecycle_Status", "Replaces_SKU", "Launch_Date", "EOL_Date",
                 "Stockout_Flag", "Demand_Imputed_Flag", "Negative_Units_Flag"]
    panel = panel.merge(hist[hist_cols], on=KEYS, how="left", suffixes=("", "_hist"))
    panel["is_test"] = False
    future_meta = future[[*KEYS, "Customer_Name", "Sales_Channel", "Product_Family", "Business_Unit",
                          "SKU_Description", "Size_Class", "Powertrain_Application", "Lifecycle_Status",
                          "Replaces_SKU", "Launch_Date", "EOL_Date"]].copy()
    future_meta["is_test"] = True
    panel = panel.merge(future_meta, on=KEYS, how="left", suffixes=("", "_test"))
    panel["is_test"] = panel["is_test_test"].fillna(False) if "is_test_test" in panel.columns else panel["is_test"]
    if "is_test_test" in panel.columns:
        panel.drop(columns="is_test_test", inplace=True)
    for c in ["Customer_Name", "Sales_Channel", "Product_Family", "Business_Unit", "SKU_Description",
              "Size_Class", "Powertrain_Application", "Lifecycle_Status", "Replaces_SKU", "Launch_Date", "EOL_Date"]:
        test_c = f"{c}_test"
        if test_c in panel.columns:
            panel[c] = panel[test_c].combine_first(panel[c])
            panel.drop(columns=test_c, inplace=True)
    # Recover static master fields for series with no transaction history.
    master_cols = ["SKU_ID", "Product_Family", "Business_Unit", "SKU_Description", "Size_Class",
                   "Powertrain_Application", "Launch_Date", "EOL_Date", "Replaces_SKU", "Lifecycle_Status"]
    master = sku[master_cols].drop_duplicates("SKU_ID")
    for c in master_cols[1:]:
        m = master.set_index("SKU_ID")[c]
        panel[c] = panel[c].combine_first(panel["SKU_ID"].map(m))
    # Forward-fill only known historical metadata within a series; test dimensions stay authoritative.
    panel = panel.sort_values(["series_id", "Date"]).reset_index(drop=True)
    for c in ["Customer_Name", "Sales_Channel"]:
        panel[c] = panel.groupby("series_id")[c].ffill().fillna("Unknown")
    for c in ["Product_Family", "Business_Unit", "SKU_Description", "Size_Class", "Powertrain_Application", "Lifecycle_Status", "Replaces_SKU"]:
        panel[c] = _clean_text(panel[c])
    panel["Replaces_SKU"] = panel["Replaces_SKU"].replace({"Unknown": "None"})
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel["Year"] = panel["Date"].dt.year
    panel["Month"] = panel["Date"].dt.month
    panel["Quarter"] = panel["Date"].dt.quarter
    panel["Demand_Target"] = pd.to_numeric(panel["Demand_Target"], errors="coerce")
    # Add stockout requested demand where the ERP row itself was absent.
    s = stock[[*KEYS, "Requested_Units"]].copy()
    s["Date"] = pd.to_datetime(s["Date"])
    panel = panel.merge(s, on=KEYS, how="left")
    panel["Demand_Target"] = panel["Requested_Units"].combine_first(panel["Demand_Target"])
    panel.drop(columns="Requested_Units", inplace=True)
    panel["is_test"] = panel["is_test"].fillna(False).astype(bool)

    # Macro and event drivers are available for both history and the forecast horizon.
    panel = panel.merge(macro, on="Date", how="left")
    panel = _apply_event_features(panel, events)
    panel["Launch_Date"] = pd.to_datetime(panel["Launch_Date"], errors="coerce")
    panel["EOL_Date"] = pd.to_datetime(panel["EOL_Date"], errors="coerce")
    panel["Months_Since_Launch"] = _month_diff(panel["Date"], panel["Launch_Date"]).fillna(-1).clip(lower=-1)
    panel["Months_To_EOL"] = _month_diff(panel["EOL_Date"], panel["Date"]).fillna(999).clip(lower=-999)
    panel["Is_Pre_Launch"] = (panel["Launch_Date"].notna() & (panel["Date"] < panel["Launch_Date"])).astype(int)
    panel["Is_Post_EOL"] = (panel["EOL_Date"].notna() & (panel["Date"] > panel["EOL_Date"])).astype(int)
    panel["month_sin"] = np.sin(2 * np.pi * panel["Month"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["Month"] / 12)
    panel["trend_index"] = ((panel["Year"] - panel["Year"].min()) * 12 + panel["Month"]).astype(float)
    return panel, future_meta


def _add_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    x = panel.sort_values(["series_id", "Date"]).copy()
    grouped = x.groupby("series_id", sort=False)["Demand_Target"]
    for lag in [1, 2, 3, 6, 12, 18, 24]:
        x[f"lag_{lag}"] = grouped.shift(lag)
    shifted = x.groupby("series_id", sort=False)["Demand_Target"].shift(1)
    for w in [3, 6, 12]:
        x[f"rolling_mean_{w}"] = shifted.groupby(x["series_id"], sort=False).transform(
            lambda s, win=w: s.rolling(win, min_periods=1).mean()
        )
    x["rolling_median_12"] = shifted.groupby(x["series_id"], sort=False).transform(
        lambda s: s.rolling(12, min_periods=1).median()
    )
    x["rolling_std_6"] = shifted.groupby(x["series_id"], sort=False).transform(
        lambda s: s.rolling(6, min_periods=2).std()
    )
    zero_flag = shifted.eq(0).where(shifted.notna())
    x["zero_rate_12"] = zero_flag.groupby(x["series_id"], sort=False).transform(
        lambda s: s.rolling(12, min_periods=1).mean()
    )
    obs_flag = shifted.notna().astype(float)
    x["observations_12"] = obs_flag.groupby(x["series_id"], sort=False).transform(
        lambda s: s.rolling(12, min_periods=1).sum()
    )
    x["expanding_mean"] = shifted.groupby(x["series_id"], sort=False).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    x["observations_to_date"] = shifted.groupby(x["series_id"], sort=False).transform(
        lambda s: s.notna().expanding(min_periods=1).sum()
    )
    return x


def _feature_columns() -> list[str]:
    macro_features = MACRO_COLS + [f"{c}_lag{k}" for c in MACRO_COLS for k in [1, 3, 6]]
    numeric = ["Year", "Month", "Quarter", "month_sin", "month_cos", "trend_index",
               "Months_Since_Launch", "Months_To_EOL", "Is_Pre_Launch", "Is_Post_EOL",
               *LAG_COLS, *ROLL_COLS, "observations_to_date", *macro_features, *EVENT_COLS]
    return CAT_COLS + numeric


def _prepare_model_matrix(frame: pd.DataFrame, categories: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    cols = _feature_columns()
    X = frame[cols].copy()
    category_map: dict[str, list[str]] = {} if categories is None else categories
    for c in CAT_COLS:
        X[c] = X[c].fillna("Unknown").astype(str)
        if categories is None:
            category_map[c] = sorted(set(X[c].tolist()) | {"Unknown"})
        X[c] = pd.Categorical(X[c], categories=category_map[c])
    for c in X.columns:
        if c not in CAT_COLS:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, category_map


def _wape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    denom = np.abs(a).sum()
    return float(np.abs(a - p).sum() / denom) if denom else float("nan")


def _bias(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    denom = np.abs(a).sum()
    return float((p - a).sum() / denom) if denom else float("nan")


def _baseline_predictions(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fallback = rows["rolling_mean_12"].fillna(rows["rolling_mean_6"]).fillna(rows["expanding_mean"]).fillna(0)
    seasonal = rows["lag_12"].fillna(rows["lag_1"]).fillna(fallback)
    recent = rows["rolling_mean_3"].fillna(rows["rolling_mean_6"]).fillna(fallback)
    return np.maximum(seasonal.to_numpy(dtype=float), 0), np.maximum(recent.to_numpy(dtype=float), 0)


def _model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1", n_estimators=700, learning_rate=0.035,
        num_leaves=31, max_depth=-1, min_child_samples=35,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        reg_alpha=0.05, reg_lambda=2.0, n_jobs=-1,
        random_state=42, verbosity=-1,
    )


def _make_anomaly_register(clean: pd.DataFrame, raw_prepared: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    x = clean.copy()
    x["Date"] = pd.to_datetime(x["Date"] + "-01")
    q999 = float(x["Demand_Target"].quantile(0.999))
    # Robust, non-destructive flag: outliers remain in model training.
    q995_by = x.groupby(["SKU_ID", "Region"])["Demand_Target"].transform(lambda s: s.quantile(0.995) if len(s) >= 12 else np.inf)
    extreme = x["Demand_Target"].gt(np.maximum(q999, q995_by * 3))
    records = []
    base_cols = ["Date", "SKU_ID", "Region", "Customer_Name", "Demand_Target", "Units_Sold_Clean"]
    for _, r in x.loc[extreme, base_cols].iterrows():
        records.append({**r.to_dict(), "Anomaly_Type": "Extreme demand", "Reason": f"Demand exceeds robust threshold (global 99.9th percentile ≈ {q999:.1f}).", "Recommended_Action": "Keep the observation; verify against customer orders and known events."})
    for _, r in x.loc[x["Negative_Units_Flag"].fillna(False), base_cols].iterrows():
        records.append({**r.to_dict(), "Anomaly_Type": "Negative quantity", "Reason": "Negative ERP quantity is not a valid demand count.", "Recommended_Action": "Set shipment quantity to zero and retain an audit flag."})
    for _, r in x.loc[x["Demand_Imputed_Flag"].fillna(False), base_cols].iterrows():
        records.append({**r.to_dict(), "Anomaly_Type": "Missing quantity imputed", "Reason": "Units_Sold was missing on an existing ERP row.", "Recommended_Action": "Review the source ERP transaction; modeling value uses SKU-region-month median fallback."})
    for _, r in x.loc[x["Stockout_Flag"].fillna(False), base_cols].iterrows():
        records.append({**r.to_dict(), "Anomaly_Type": "Stockout demand correction", "Reason": "Requested demand is greater than shipped units in the stockout file.", "Recommended_Action": "Use Requested_Units as unconstrained demand for forecasting."})
    out = pd.DataFrame(records)
    if not out.empty:
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m")
        out = out.sort_values(["Date", "Anomaly_Type", "SKU_ID", "Region"])
    return out, int(extreme.sum())


def _recursive_forecast(panel: pd.DataFrame, X_categories: dict[str, list[str]], model: LGBMRegressor,
                        test_keys: pd.DataFrame, sku: pd.DataFrame) -> pd.DataFrame:
    """Forecast test rows month-by-month, updating lag history with each month's predictions."""
    xpanel = panel.sort_values(["series_id", "Date"]).copy()
    max_train_date = xpanel.loc[~xpanel["is_test"], "Date"].max()
    # A full month-indexed history is maintained for every SKU-region series.
    hist_map: dict[int, list[float]] = {}
    for sid, g in xpanel[xpanel["Date"] <= max_train_date].groupby("series_id", sort=False):
        hist_map[int(sid)] = g["Demand_Target"].astype(float).tolist()
    future_dates = sorted(xpanel.loc[xpanel["is_test"], "Date"].unique())
    master = sku.set_index("SKU_ID")
    output = []

    def stats_from_history(h: list[float]) -> dict[str, float]:
        values = np.asarray(h, dtype=float)
        result: dict[str, float] = {}
        for lag in [1, 2, 3, 6, 12, 18, 24]:
            result[f"lag_{lag}"] = float(values[-lag]) if len(values) >= lag else np.nan
        for w in [3, 6, 12]:
            tail = values[-w:] if len(values) else values
            valid = tail[np.isfinite(tail)]
            result[f"rolling_mean_{w}"] = float(valid.mean()) if len(valid) else np.nan
        tail12 = values[-12:] if len(values) else values
        valid12 = tail12[np.isfinite(tail12)]
        result["rolling_median_12"] = float(np.median(valid12)) if len(valid12) else np.nan
        tail6 = values[-6:] if len(values) else values
        valid6 = tail6[np.isfinite(tail6)]
        result["rolling_std_6"] = float(np.std(valid6, ddof=1)) if len(valid6) >= 2 else np.nan
        result["zero_rate_12"] = float(np.mean(valid12 == 0)) if len(valid12) else np.nan
        result["observations_12"] = float(len(valid12))
        all_valid = values[np.isfinite(values)]
        result["expanding_mean"] = float(all_valid.mean()) if len(all_valid) else np.nan
        result["observations_to_date"] = float(len(all_valid))
        return result

    # Pre-compute SKU -> predecessor mapping for cold-start launches.
    replaced_by = master["Replaces_SKU"].to_dict() if "Replaces_SKU" in master.columns else {}
    sid_lookup = {(r.SKU_ID, r.Region): int(r.series_id) for r in xpanel[["SKU_ID", "Region", "series_id"]].drop_duplicates().itertuples()}

    for month in future_dates:
        month_rows = xpanel[(xpanel["Date"] == month) & xpanel["is_test"]].copy()
        if month_rows.empty:
            # Preserve the monthly position for every series if there is a gap in test coverage.
            for sid in hist_map:
                hist_map[sid].append(np.nan)
            continue
        feature_rows = month_rows.copy()
        for idx, row in feature_rows.iterrows():
            feature_stats = stats_from_history(hist_map.get(int(row["series_id"]), []))
            for col, val in feature_stats.items():
                feature_rows.at[idx, col] = val
        X, _ = _prepare_model_matrix(feature_rows, categories=X_categories)
        prediction = np.expm1(model.predict(X))
        prediction = np.maximum(prediction, 0)

        # Product lifecycle hard constraints; for cold-start replacements, transfer predecessor run-rate.
        for j, (_, row) in enumerate(feature_rows.iterrows()):
            if row["Is_Pre_Launch"] == 1 or row["Is_Post_EOL"] == 1:
                prediction[j] = 0.0
                continue
            sid = int(row["series_id"])
            hist = np.asarray(hist_map.get(sid, []), dtype=float)
            if np.isfinite(hist).sum() == 0:
                predecessor = replaced_by.get(row["SKU_ID"], "None")
                pred_sid = sid_lookup.get((predecessor, row["Region"])) if predecessor not in {None, "None", "Unknown"} else None
                if pred_sid is not None and pred_sid in hist_map:
                    prior = np.asarray(hist_map[pred_sid], dtype=float)
                    prior = prior[np.isfinite(prior)][-12:]
                    if len(prior):
                        prediction[j] = float(np.median(prior))

        month_prediction = {}
        for (_, row), pred in zip(month_rows.iterrows(), prediction):
            sid = int(row["series_id"])
            month_prediction[sid] = float(pred)
            output.append({"Date": pd.Timestamp(month).strftime("%Y-%m"), "SKU_ID": row["SKU_ID"], "Region": row["Region"],
                           "Units_Sold_Forecast": int(max(0, np.rint(pred)))})
        # Append an observed prediction (or NaN for absent keys) for every series.
        for sid in hist_map:
            hist_map[sid].append(month_prediction.get(sid, np.nan))

    forecast = pd.DataFrame(output)
    # Standardize Date datatype before merging
    test_keys = test_keys.copy()
    forecast = forecast.copy()

    test_keys["Date"] = pd.to_datetime(test_keys["Date"])
    forecast["Date"] = pd.to_datetime(forecast["Date"])
    # Preserve exactly the order and number of rows in test.csv.
    forecast = test_keys[[*KEYS]].merge(forecast, on=KEYS, how="left")
    forecast["Units_Sold_Forecast"] = forecast["Units_Sold_Forecast"].fillna(0).clip(lower=0).round().astype(int)
    return forecast


def run_pipeline() -> dict[str, Any]:
    print("[1/8] Loading input files...")
    inp = _load_inputs()
    sku = _prepare_sku_master(inp["sku"])
    stock = _prepare_stockouts(inp["stockouts"])
    macro = _prepare_macro(inp["macro"])
    events = _prepare_events(inp["events"])

    print("[2/8] Cleaning training data...")
    clean, clean_metrics, raw_prepared = _prepare_train(inp["train"], sku, stock)
    test = _prepare_test(inp["test"], sku)
    clean.to_csv(OUT / "cleaned_history.csv", index=False)

    print("[3/8] Building monthly panel and business-driver features...")
    panel, test_meta = _build_panel(clean, test, sku, stock, macro, events)
    panel = _add_history_features(panel)
    history_rows = panel[
    (~panel["is_test"]) & panel["Demand_Target"].notna()
].copy()

    history_end = history_rows["Date"].max()

    print("[4/8] Time-based backtest (last 12 months)...")

    validation_start = history_end - pd.DateOffset(months=11)

    train_rows = history_rows[
    history_rows["Date"] < validation_start
].copy()

    valid_rows = history_rows[
    history_rows["Date"] >= validation_start
].copy()

    print("\n========== VALIDATION CHECK ==========")
    print("History end:", history_end)
    print("Validation start:", validation_start)
    print("Training rows:", train_rows.shape)
    print("Validation rows:", valid_rows.shape)
    print("======================================\n")

    if train_rows.empty or valid_rows.empty:
        raise ValueError(
        "Training or validation dataset is empty. Check the date range."
    )

    train_idx = train_rows.index
    valid_idx = valid_rows.index
    X_all, categories = _prepare_model_matrix(panel)

    y_train = np.log1p(
    train_rows["Demand_Target"].astype(float).clip(lower=0)
)

    y_valid = valid_rows["Demand_Target"].astype(float).clip(lower=0)

    X_train = X_all.loc[train_idx]
    X_valid = X_all.loc[valid_idx]

    print("\n========== DEBUG INFORMATION ==========")
    print("History rows:", history_rows.shape)
    print("Training rows:", train_rows.shape)
    print("Validation rows:", valid_rows.shape)
    print("Complete feature matrix:", X_all.shape)
    print("Training feature matrix:", X_train.shape)
    print("Validation feature matrix:", X_valid.shape)
    print("Feature columns:", list(X_all.columns))
    print("=======================================\n")

    model_bt = _model()
    model_bt.fit(
        X_train, y_train,
        eval_set=[(X_valid, np.log1p(y_valid))],
        eval_metric="l1",
        callbacks=[early_stopping(stopping_rounds=70, verbose=False), log_evaluation(period=0)],
        categorical_feature=CAT_COLS,
    )
    bt_pred = np.maximum(np.expm1(model_bt.predict(X_valid)), 0)
    base_seasonal, base_recent = _baseline_predictions(valid_rows)
    bt = valid_rows[["Date", "SKU_ID", "Region", "Product_Family", "Sales_Channel", "Customer_Name", "Demand_Target"]].copy()
    bt["Baseline_Seasonal"] = base_seasonal
    bt["Baseline_Recent"] = base_recent
    bt["Model_Forecast"] = bt_pred
    bt["Horizon_Month"] = ((bt["Date"].dt.year - validation_start.year) * 12 + bt["Date"].dt.month - validation_start.month + 1).clip(lower=1)
    bt["Date"] = bt["Date"].dt.strftime("%Y-%m")
    bt.to_csv(OUT / "backtest_predictions.csv", index=False)

    model_metrics = {
        "validation_start": validation_start.strftime("%Y-%m"),
        "validation_end": history_end.strftime("%Y-%m"),
        "validation_rows": int(len(valid_rows)),
        "best_iteration": int(model_bt.best_iteration_ or model_bt.n_estimators),
        "baseline_seasonal_wape": _wape(y_valid, base_seasonal),
        "baseline_seasonal_bias": _bias(y_valid, base_seasonal),
        "baseline_recent_wape": _wape(y_valid, base_recent),
        "baseline_recent_bias": _bias(y_valid, base_recent),
        "lightgbm_wape": _wape(y_valid, bt_pred),
        "lightgbm_bias": _bias(y_valid, bt_pred),
    }
    print("    Backtest metrics:", json.dumps(model_metrics, indent=2))

    # Slice metrics by family, region, and horizon, as required by the brief.
    slices = []
    for dimension in ["Product_Family", "Region", "Horizon_Month"]:
        for key, g in bt.groupby(dimension, dropna=False):
            slices.append({"Dimension": dimension, "Segment": str(key), "Rows": int(len(g)),
                           "WAPE": _wape(g["Demand_Target"], g["Model_Forecast"]),
                           "Bias": _bias(g["Demand_Target"], g["Model_Forecast"])})
    pd.DataFrame(slices).to_csv(OUT / "backtest_metrics_by_segment.csv", index=False)

    print("[5/8] Fitting final model on all available history...")
    all_rows = history_rows
    X_fit = X_all.loc[all_rows.index]
    y_fit = np.log1p(all_rows["Demand_Target"].astype(float).clip(lower=0))
    final_model = _model()
    final_model.set_params(n_estimators=max(100, int(model_metrics["best_iteration"])))
    final_model.fit(X_fit, y_fit, categorical_feature=CAT_COLS)

    print("[6/8] Producing recursive 19-month forecast...")
    test_keys = test[[*KEYS]].copy()
    forecast = _recursive_forecast(panel, categories, final_model, test_keys, sku)
    forecast.to_csv(OUT / "forecast.csv", index=False)

    # Submission must match every row in test.csv and the requested column order.
    submission = inp["sample"].copy()
    submission["Date"] = submission["Date"].astype(str)
    full_test_keys = test[[*KEYS]].copy()
    full_test_keys["Date"] = full_test_keys["Date"].dt.strftime("%Y-%m")
    forecast_join = forecast.copy()
# Standardize Date datatype before final submission merge
    full_test_keys = full_test_keys.copy()
    forecast_join = forecast_join.copy()

    full_test_keys["Date"] = pd.to_datetime(full_test_keys["Date"])
    forecast_join["Date"] = pd.to_datetime(forecast_join["Date"])

    submission = full_test_keys.merge(
    forecast_join,
    on=KEYS,
    how="left"
)[["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]]    
    submission["Units_Sold_Forecast"] = submission["Units_Sold_Forecast"].fillna(0).astype(int)
    submission.to_csv(OUT / "sample_submission_completed.csv", index=False)

    print("[7/8] Creating anomaly register and data-quality report...")
    anomaly, extreme_count = _make_anomaly_register(clean, raw_prepared)
    anomaly.to_csv(OUT / "anomaly_register.csv", index=False)
    dq_rows = [
        {"File": "train.csv", "Issue": "Raw rows", "Rows_Affected": clean_metrics["train_rows_raw"], "Action": "Loaded for profiling."},
        {"File": "train.csv", "Issue": "Invalid dates removed", "Rows_Affected": clean_metrics["invalid_date_rows_dropped"], "Action": "Rows without a parseable authoritative Date were excluded."},
        {"File": "train.csv", "Issue": "Repeated Record_ID rows", "Rows_Affected": clean_metrics["duplicate_record_id_rows_removed"], "Action": "Kept row with latest Load_Timestamp."},
        {"File": "train.csv", "Issue": "Duplicate Date-SKU-Region rows", "Rows_Affected": clean_metrics["duplicate_key_rows_aggregated"], "Action": "Aggregated valid transaction quantities to one monthly key."},
        {"File": "train.csv", "Issue": "Date vs Year/Month disagreement", "Rows_Affected": clean_metrics["year_month_disagreement_rows"], "Action": "Trusted Date and recomputed Year and Month."},
        {"File": "train.csv", "Issue": "Missing Units_Sold", "Rows_Affected": clean_metrics["missing_units_rows"], "Action": "Used stockout requested demand where available; otherwise seasonal/series median imputation."},
        {"File": "train.csv", "Issue": "Non-numeric Units_Sold", "Rows_Affected": clean_metrics["non_numeric_units_rows"], "Action": "Parsed comma-formatted values; unresolved values treated as missing then imputed."},
        {"File": "train.csv", "Issue": "Negative Units_Sold", "Rows_Affected": clean_metrics["negative_units_rows"], "Action": "Clipped invalid negative demand to zero and flagged it."},
        {"File": "inventory_stockouts.csv", "Issue": "Matched stockout keys", "Rows_Affected": clean_metrics["stockout_rows_matched"], "Action": "Used Requested_Units as unconstrained demand."},
        {"File": "inventory_stockouts.csv", "Issue": "Demand uplift from stockouts", "Rows_Affected": round(clean_metrics["stockout_demand_uplift_units"], 2), "Action": "Added requested-minus-shipped demand into training target."},
        {"File": "train.csv", "Issue": "Zero or invalid unit prices", "Rows_Affected": clean_metrics["invalid_or_zero_price_rows"], "Action": "Imputed SKU-year median positive price, then SKU median."},
        {"File": "cleaned_history.csv", "Issue": "Extreme demand flags", "Rows_Affected": extreme_count, "Action": "Retained observations; flagged for business validation."},
    ]
    dq = pd.DataFrame(dq_rows)
    dq.to_csv(OUT / "data_quality_report.csv", index=False)

    print("[8/8] Writing metrics and metadata...")
    result = {
        "project": "Celeris Automobili Industrial - Demand Forecasting & Planning Control Tower",
        "history_start": pd.to_datetime(clean["Date"] + "-01").min().strftime("%Y-%m"),
        "history_end": history_end.strftime("%Y-%m"),
        "forecast_start": test["Date"].min().strftime("%Y-%m"),
        "forecast_end": test["Date"].max().strftime("%Y-%m"),
        "train_clean_rows": int(len(clean)),
        "test_rows": int(len(test)),
        "series_count": int(test[["SKU_ID", "Region"]].drop_duplicates().shape[0]),
        "forecast_total_units": int(forecast["Units_Sold_Forecast"].sum()),
        "forecast_mean_units": float(forecast["Units_Sold_Forecast"].mean()),
        "anomaly_register_rows": int(len(anomaly)),
        "extreme_demand_rows": int(extreme_count),
        "cleaning": clean_metrics,
        "backtest": model_metrics,
    }
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    # Save feature importance for model explainability.
    importance = pd.DataFrame({"Feature": _feature_columns(), "Importance": final_model.feature_importances_})
    importance.sort_values("Importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)

    print("\nPIPELINE COMPLETE")
    print(f"Forecast rows: {len(forecast):,}; expected test rows: {len(test):,}")
    print(f"Forecast total units: {forecast['Units_Sold_Forecast'].sum():,}")
    print(f"Outputs saved to: {OUT}")
    return result


if __name__ == "__main__":
    run_pipeline()
