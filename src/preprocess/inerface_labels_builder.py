from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import interface_data_structs as ids


@dataclass
class LabelsBuilderOutputs:
    top_n_diagnoses_pickle: Path


class AbstractLabelsBuilder(ABC):
    @abstractmethod
    def process(
        self,
            filtered_diagnoses_icd_pickle: Path,
            d_icd_diagnoses_csv: Path
    ) -> LabelsBuilderOutputs:
        pass
