#  TODO change filename since not all classes are strictly inputs
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SQL_OUTPUT_DIR = DATA_DIR / "mimiciii_query_results"
PREFILTER_OUTPUT_DIR = DATA_DIR / "output_prefilter"
MERGED_STAY_MEASUREMENT_OUTPUT_DIR = (
    DATA_DIR / "output_merged_stay_measurements"
)

SAMPLE_LIST_BUILDER_OUTPUT_DIR = (
    DATA_DIR / "output_full_admission_list_builder"
)

FEATURE_BUILDER_OUTPUT_DIR = DATA_DIR / "output_feature_builder"

FEATURE_FINALIZER_OUTPUT_DIR = DATA_DIR / "output_feature_finalizer"

BG_DATA_COLS = ["potassium", "calcium", "ph", "pco2", "lactate"]
LAB_DATA_COLS = [
    "albumin",
    "bun",
    "creatinine",
    "sodium",
    "bicarbonate",
    "platelet",
    "glucose",
    "inr",
]
VITAL_DATA_COLS = [
    "heartrate",
    "sysbp",
    "diasbp",
    "tempc",
    "resprate",
    "spo2",
]


@dataclass
class PrefilterResourceRefs:
    icustay: Path = SQL_OUTPUT_DIR / "icustay_detail.csv"
    bg: Path = SQL_OUTPUT_DIR / "pivoted_bg.csv"
    vital: Path = SQL_OUTPUT_DIR / "pivoted_vital.csv"
    lab: Path = SQL_OUTPUT_DIR / "pivoted_lab.csv"


@dataclass
class PrefilterSettings:
    output_dir: Path = PREFILTER_OUTPUT_DIR
    min_age: int = 18
    min_los_hospital: int = 1
    min_los_icu: int = 1
    bg_data_cols: list[str] = None
    lab_data_cols: list[str] = None
    vital_data_cols: list[str] = None

    def __post_init__(self):
        if self.bg_data_cols is None:
            self.bg_data_cols = BG_DATA_COLS
        if self.lab_data_cols is None:
            self.lab_data_cols = LAB_DATA_COLS
        if self.vital_data_cols is None:
            self.vital_data_cols = VITAL_DATA_COLS


@dataclass
class ICUStayMeasurementCombinerResourceRefs:
    icustay: Path = PREFILTER_OUTPUT_DIR / "icustay.pickle"
    bg: Path = PREFILTER_OUTPUT_DIR / "bg.pickle"
    lab: Path = PREFILTER_OUTPUT_DIR / "lab.pickle"
    vital: Path = PREFILTER_OUTPUT_DIR / "vital.pickle"


@dataclass
class ICUStayMeasurementCombinerSettings:
    output_dir: Path = MERGED_STAY_MEASUREMENT_OUTPUT_DIR
    winsorize_upper: float = 0.95
    winsorize_lower: float = 0.05
    bg_data_cols: list[str] = None
    lab_data_cols: list[str] = None
    vital_data_cols: list[str] = None

    def __post_init__(self):
        if self.bg_data_cols is None:
            self.bg_data_cols = BG_DATA_COLS
        if self.lab_data_cols is None:
            self.lab_data_cols = LAB_DATA_COLS
        if self.vital_data_cols is None:
            self.vital_data_cols = VITAL_DATA_COLS

    @property
    def all_measurement_cols(self) -> list[str]:
        return self.bg_data_cols + self.lab_data_cols + self.vital_data_cols


@dataclass
class FullAdmissionListBuilderResourceRefs:
    icustay_bg_lab_vital: Path = (
        MERGED_STAY_MEASUREMENT_OUTPUT_DIR / "icustay_bg_lab_vital.pickle"
    )


@dataclass
class FullAdmissionListBuilderSettings:
    output_dir: Path = SAMPLE_LIST_BUILDER_OUTPUT_DIR
    measurement_cols: list[str] = None

    def __post_init__(self):
        if self.measurement_cols is None:
            self.measurement_cols = (
                BG_DATA_COLS + LAB_DATA_COLS + VITAL_DATA_COLS
            )

    @property
    def time_series_cols(self) -> list[str]:
        return ["charttime"] + self.measurement_cols


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


#  https://stackoverflow.com/a/65392400  (need this to work with dill)
FullAdmissionData.__module__ = __name__


@dataclass
class FeatureBuilderResourceRefs:
    full_admission_list: Path = (
        SAMPLE_LIST_BUILDER_OUTPUT_DIR / "full_admission_list.pickle"
    )
    bg_lab_vital_summary_stats: Path = (
        MERGED_STAY_MEASUREMENT_OUTPUT_DIR
        / "bg_lab_vital_summary_stats.pickle"
    )


@dataclass
class FeatureBuilderSettings:
    output_dir: Path = FEATURE_BUILDER_OUTPUT_DIR
    winsorize_low: str = "5%"
    winsorize_high: str = "95%"
    resample_interpolation_method: str = "linear"
    resample_limit_direction: str = "both"


#     TODO add data member for cutoff time after admit or before discharge


@dataclass
class FeatureFinalizerResourceRefs:
    processed_admission_list: Path = (
        FEATURE_BUILDER_OUTPUT_DIR / "hadm_list_with_processed_dfs.pickle"
    )


@dataclass
class FeatureFinalizerSettings:
    output_dir: Path = FEATURE_FINALIZER_OUTPUT_DIR
    observation_window_in_hours: int = 48
    require_exact_num_hours: bool = True  # when True, no need for padding
    observation_window_start: str = "intime"
