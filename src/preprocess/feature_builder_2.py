import numpy as np
import dill
import pickle
import pandas as pd
import scipy.stats.mstats as mstats
from dataclasses import dataclass
from pathlib import Path
import resource_io as rio
from prefilter_input_classes import (
    BG_DATA_COLS,
    LAB_DATA_COLS,
    VITAL_DATA_COLS,
)


def winsorize(
    df: pd.DataFrame,
    cols: list[str],
    lower_cutoff: float,
    upper_cutoff: float,
):
    q_lower = df[cols].quantile(q=lower_cutoff)
    q_upper = df[cols].quantile(q=upper_cutoff)
    df[cols] = df[cols].clip(lower=q_lower, upper=q_upper, axis=1)


my_importer = rio.ResourceImporter()
stay_meas_data = my_importer.import_resource(
    resource=Path(
        "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack"
        "/data/output_merged_stay_measurements"
        "/icustay_bg_lab_vital.pickle"
    )
)

stay_meas_data = pd.DataFrame(stay_meas_data)
stay_meas_data = stay_meas_data.drop(
    [
        "dod",
        "los_hospital",
        "admission_age",
        "hospstay_seq",
        "icustay_seq",
        "first_hosp_stay",
        "los_icu",
        "first_icu_stay",
        # "icustay_id_bg",
        # "icustay_id_lab",
    ],
    axis=1,
)

winsorize(
    df=stay_meas_data,
    cols=BG_DATA_COLS + LAB_DATA_COLS + VITAL_DATA_COLS,
    lower_cutoff=0.05,
    upper_cutoff=0.95,
)

groups = stay_meas_data.groupby(["hadm_id"])
list_of_dfs = [group[1] for group in groups]
time_series_cols = (
    ["charttime"] + BG_DATA_COLS + LAB_DATA_COLS + VITAL_DATA_COLS
)


@dataclass
class FullAdmissionData:
    subject_id: np.ndarray
    hadm_id: np.ndarray
    icustay_id: np.ndarray
    admittime: np.ndarray
    dischtime: np.ndarray
    hospital_expire_flag: np.ndarray
    intime: np.ndarray
    outtime: np.ndarray
    time_series: pd.DataFrame


#  https://stackoverflow.com/a/65392400
FullAdmissionData.__module__ = __name__

list_of_full_admission_data = [
    FullAdmissionData(
        subject_id=np.unique(item.subject_id),
        hadm_id=np.unique(item.hadm_id),
        icustay_id=np.unique(item.icustay_id),
        admittime=np.unique(item.admittime),
        dischtime=np.unique(item.dischtime),
        hospital_expire_flag=np.unique(item.hospital_expire_flag),
        intime=np.unique(item.intime),
        outtime=np.unique(item.outtime),
        time_series=item[time_series_cols],
    )
    for item in list_of_dfs
]

output_path = Path(
    "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
    "/output_full_admission_objects/full_admisssion.pickle"
)

dill.dump(list_of_full_admission_data, output_path.open(mode="wb"))
