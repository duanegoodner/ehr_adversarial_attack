from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class PrefilterResourceRefs:
    # admissions: pr.PreprocessResource
    d_icd_diagnoses: Path
    diagnoses_icd: Path
    icustay_detail: Path
    # pivoted_bg: pr.PreprocessResource
    # pivoted_gc: pr.PreprocessResource
    # pivoted_lab: pr.PreprocessResource
    # pivoted_uo: pr.PreprocessResource
    # pivoted_vital: pr.PreprocessResource


@dataclass
class PrefilterResources:
    d_icd_diagnoses: pd.DataFrame
    diagnoses_icd: pd.DataFrame
    icustay_detail: pd.DataFrame


@dataclass
class PrefilterSettings:
    output_dir: Path
    min_age: int = 18
    min_los_hospital: int = 1
    min_los_icu: int = 1