import dill
import numpy as np
import pandas as pd

import resource_io as rio
from pathlib import Path
from prefilter_input_classes import (
    BG_DATA_COLS,
    LAB_DATA_COLS,
    VITAL_DATA_COLS,
)


def winsorize(
    df: pd.DataFrame, upper_quantile: float, lower_quantile: float
) -> pd.DataFrame:
    q_low = df.quantile(lower_quantile)
    q_high = df.quantile(upper_quantile)


full_admission_list_path = Path(
    "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
    "/output_full_admission_list_builder/full_admission_list.pickle"
)

with full_admission_list_path.open(mode="rb") as f:
    full_admission_list = dill.load(f)

stats_summary_path = Path(
    "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
    "/output_merged_stay_measurements/bg_lab_vital_summary_stats.pickle"
)

with stats_summary_path.open(mode="rb") as f:
    stats_summary = dill.load(f)


df = full_admission_list[0].time_series
df = df.sort_values("charttime")
df.set_index(["charttime"], inplace=True)
df_resampled = (
    df.resample("H")
    .mean()
    .interpolate(method="linear", limit_direction="both")
)

all_nan_cols = df_resampled.columns[df_resampled.isna().all()]
fill_na_map = {col: stats_summary.loc["50%", col] for col in all_nan_cols}
df_resampled.fillna(fill_na_map, inplace=True)

df_winsorized = df_resampled[stats_summary.columns].clip(
    lower=stats_summary.loc["5%", :],
    upper=stats_summary.loc["95%", :],
    axis=1,
)

df_normalized = (df_winsorized - stats_summary.loc["5%", :]) / (
    stats_summary.loc["95%", :] - stats_summary.loc["5%", :]
)
