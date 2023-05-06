from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class PrefilterResourceRefs:
    icustay: Path
    bg: Path
    vital: Path
    lab: Path


@dataclass
class PrefilterResources:
    icustay: pd.DataFrame
    bg: pd.DataFrame
    vital: pd.DataFrame
    lab: pd.DataFrame


@dataclass
class PrefilterSettings:
    output_dir: Path
    min_age: int = 18
    min_los_hospital: int = 1
    min_los_icu: int = 1


@dataclass
class ICUStayMeasurementCombinerResourceRefs:
    icustay: Path
    bg: Path
    lab: Path
    vital: Path


@dataclass
class ICUStayMeasurementCombinerResources:
    icustay: Path
    bg: Path
    lab: Path
    vital: Path


@dataclass
class ICUStayMeasurementCombinerSettings:
    output_dir: Path
    winsorize_upper: float
    winsorize_lower: float
    bg_data_cols: list[str]
    lab_data_cols: list[str]
    vital_data_cols: list[str]

    @property
    def all_measurement_cols(self) -> list[str]:
        return self.bg_data_cols + self.lab_data_cols + self.vital_data_cols


@dataclass
class FeatureBuilderResourceRefs:
    icustay: Path
    bg: Path
    lab: Path
    vital: Path


@dataclass
class FeatureBuilderResources:
    icustay: pd.DataFrame
    bg: pd.DataFrame
    lab: pd.DataFrame
    vital: pd.DataFrame


@dataclass
class FeatureBuilderSettings:
    output_dir: Path
    winsorize_upper: float
    winsorize_lower: float
    bg_data_cols: list[str]
    lab_data_cols: list[str]
    vital_data_cols: list[str]

    @property
    def all_measurement_cols(self) -> list[str]:
        return self.bg_data_cols + self.lab_data_cols + self.vital_data_cols
