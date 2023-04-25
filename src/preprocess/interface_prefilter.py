from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrefilterSettings:
    output_dir: Path
    min_age: int = 18
    min_los_hospital: int = 1
    min_los_icu: int = 1


@dataclass
class PrefilterOutputs:
    filtered_icustay_detail_pickle: Path
    filtered_diagnoses_icd_pickle: Path
    filtered_d_diagnoses_icd_pickle: Path


class AbstractPrefilter(ABC):
    @abstractmethod
    def process(
        self,
        icustay_detail_csv: Path,
        diagnoses_icd_csv: Path,
        d_diagnoses_icd_csv: Path
    ) -> PrefilterOutputs:
        pass
