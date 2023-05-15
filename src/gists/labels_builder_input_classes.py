from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class LabelsBuilderResourceRefs:
    d_icd_diagnoses: Path
    diagnoses_icd: Path


@dataclass
class LabelsBuilderResources:
    d_icd_diagnoses: pd.DataFrame
    diagnoses_icd: pd.DataFrame


@dataclass
class LabelsBuilderSettings:
    output_dir: Path
    num_diagnoses: int







