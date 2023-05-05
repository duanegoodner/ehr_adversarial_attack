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
    icustay: pd.DataFrame = None
    bg: pd.DataFrame = None
    vital: pd.DataFrame = None
    lab: pd.DataFrame = None


@dataclass
class PrefilterSettings:
    output_dir: Path
    min_age: int = 18
    min_los_hospital: int = 1
    min_los_icu: int = 1


@dataclass
class FeatureBuilderResourceRefs:
    icustay: Path
    bg: Path
    lab: Path
    vital: Path


@dataclass
class FeatureBuilderResources:
    icustay: Path
    bg: Path
    lab: Path
    vital: Path


@dataclass
class FeatureBuilderSettings:
    output_dir: Path

