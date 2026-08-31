"""
data_prep.py
=============
Data loading & preprocessing for the CLEF (Contextualized Local Explanation
Framework) pipeline -- 2-device version (Fridge, Printer).

Pipeline (see Section 4.2 of the paper):
  1. Load the Watt (active power) file for the target appliance.
  2. Load contextual sensor files (Lab temperature, Lab humidity, and an
     occupancy/activity proxy -- see note below).
  3. Handle missing values:
       - Interior gaps of <= 2 consecutive samples -> forward-fill.
       - Leading gaps (sensor not yet active at t=0) -> backward-fill.
  4. Inner-join on the shared timestamp index (all files are recorded
     synchronously -> 361 rows, no missing values after the join).
  5. Derive Hour-of-day and Day-of-week from the timestamp.
  6. Min-max normalise all 6 numerical features to [0, 1]:
       [power, hour, day_of_week, temperature, humidity, occupancy]
  7. The context vector c_i = (hour, temperature, occupancy) is used by
     LIME-CA / context-aware SHAP for cosine-similarity based selection.

Occupancy proxy -- data-quality note (deviation from the v1 draft):
  The Lab Motion and Lab Door sensors (used for "occupancy" in the v1
  draft) were found to have ~50% missing values, with gaps of up to 21
  consecutive samples -- not recoverable by the <=2-sample forward-fill
  rule. Separately, the Kitchen and Mailroom motion logs in this dataset
  were found to be byte-for-byte identical (a labelling/export artefact in
  the source data), each with only a single missing value (>99.7% complete).
  We therefore use this well-populated series as a continuous "common-area
  activity index" (occupancy proxy), normalised to [0, 1]. The raw Lab
  Motion / Lab Door files are retained under data/raw/*_raw.csv purely for
  documentation of this data-quality issue (see Limitations).

The Alignment Score (alignment.py) uses
    F_occ = {hour, occupancy}
i.e. the two of the six input features most directly tied to human
presence / temporal behaviour -- a subset of the model's own input space
(this fixes an inconsistency in the v1 draft, where F_occ referenced 8
external sensors not present in the model's input).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

FEATURE_COLUMNS = ["power", "hour", "day_of_week", "temperature", "humidity", "occupancy"]
CONTEXT_COLUMNS = ["hour", "temperature", "occupancy"]  # context vector c_i

DEVICE_FILES = {
    "fridge": "fridge_power.csv",
    "printer": "printer_power.csv",
}


def _read_chronograf_csv(path: str, value_name: str) -> pd.DataFrame:
    """Read a single Chronograf-exported CSV (time, <value column>)."""
    df = pd.read_csv(path)
    time_col = df.columns[0]
    value_col = df.columns[1]
    df = df.rename(columns={time_col: "time", value_col: value_name})
    df["time"] = pd.to_datetime(df["time"], utc=False)
    return df[["time", value_name]]


def _fill_missing(series: pd.Series, max_interior_gap: int = 2) -> pd.Series:
    """
    Fill missing values following the rules in Section 4.2:
      - interior runs of length <= `max_interior_gap` -> forward fill
      - any remaining (typically a single *leading* run, i.e. sensor not
        yet powered on at the start of the recording window) -> backward
        fill, so that the very first observed reading is propagated
        backwards.
    """
    s = series.copy()
    # Forward-fill short interior gaps only.
    is_na = s.isna()
    # Identify run-lengths of consecutive NaNs
    groups = (~is_na).cumsum()
    run_len = is_na.groupby(groups).transform("sum")
    short_gap_mask = is_na & (run_len <= max_interior_gap)
    s[short_gap_mask] = s.ffill()[short_gap_mask]
    # Any remaining NaNs (leading run, sensor not yet active) -> bfill
    if s.isna().any():
        s = s.bfill()
    return s


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return series * 0.0
    return (series - lo) / (hi - lo)


def load_device_dataframe(device: str, normalize: bool = True) -> pd.DataFrame:
    """
    Build the 6-D feature DataFrame for `device` ('fridge' or 'printer').

    Returns a DataFrame indexed by timestamp with columns FEATURE_COLUMNS,
    plus a `raw_power` column (the un-normalised Watt reading, kept for
    reporting / sanity checks).
    """
    if device not in DEVICE_FILES:
        raise ValueError(f"Unknown device '{device}', expected one of {list(DEVICE_FILES)}")

    power = _read_chronograf_csv(os.path.join(DATA_DIR, DEVICE_FILES[device]), "power")
    temp = _read_chronograf_csv(os.path.join(DATA_DIR, "lab_temperature.csv"), "temperature")
    hum = _read_chronograf_csv(os.path.join(DATA_DIR, "lab_humidity.csv"), "humidity")
    occ = _read_chronograf_csv(os.path.join(DATA_DIR, "common_area_motion.csv"), "occupancy")

    # Fix missing values BEFORE the join (gaps occur within each sensor's
    # own series).
    power["power"] = _fill_missing(power["power"])
    temp["temperature"] = _fill_missing(temp["temperature"])
    hum["humidity"] = _fill_missing(hum["humidity"])
    occ["occupancy"] = _fill_missing(occ["occupancy"])

    # All four series share exactly the same 361 timestamps -> inner join.
    df = power.merge(temp, on="time").merge(hum, on="time").merge(occ, on="time")
    df = df.sort_values("time").reset_index(drop=True)
    assert df.isna().sum().sum() == 0, "Unexpected NaNs remain after preprocessing"

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek

    df["raw_power"] = df["power"]

    if normalize:
        for col in ["power", "hour", "day_of_week", "temperature", "humidity", "occupancy"]:
            df[col] = _minmax(df[col].astype(float))

    df = df.set_index("time")
    return df[FEATURE_COLUMNS + ["raw_power"]]


if __name__ == "__main__":
    for dev in DEVICE_FILES:
        d = load_device_dataframe(dev)
        print(dev, d.shape)
        print(d.describe().T[["min", "max", "mean", "std"]])
        print()
