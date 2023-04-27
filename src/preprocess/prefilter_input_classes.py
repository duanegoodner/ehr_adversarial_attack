from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class PrefilterResourceRefs:
    admissions: Path
    d_icd_diagnoses: Path
    diagnoses_icd: Path
    icustay_detail: Path
    pivoted_bg: Path
    pivoted_vital: Path
    pivoted_lab: Path
    # pivoted_gc: pr.PreprocessResource
    # pivoted_uo: pr.PreprocessResource


@dataclass
class PrefilterResources:
    admissions: pd.DataFrame
    d_icd_diagnoses: pd.DataFrame
    diagnoses_icd: pd.DataFrame
    icustay_detail: pd.DataFrame
    pivoted_bg: pd.DataFrame
    pivoted_vital: pd.DataFrame
    pivoted_lab: pd.DataFrame


@dataclass
class PrefilterSettings:
    output_dir: Path
    min_age: int = 18
    min_los_hospital: int = 1
    min_los_icu: int = 1