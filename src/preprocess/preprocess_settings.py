from dataclasses import dataclass
from pathlib import Path
import interface_data_structs as ids
import interface_prefilter as ip
import preprocess_resource as ppr
import preprocess_module as ppm
import prefilter_input_classes as pfin


class PreprocessSettings:
    def __init__(self, project_root: Path):
        self._project_root = project_root
        self._data_dir = project_root / "data"
        self._sql_output_dir = self._data_dir / "mimiciii_query_results"
        self._prefilter_settings = pfin.PrefilterSettings(
            output_dir=self._data_dir / "prefilter_output",
            min_age=18,
            min_los_hospital=1,
            min_los_icu=1
        ),
        self._prefilter_resource_refs = pfin.PrefilterResourceRefs(
            d_icd_diagnoses=self._sql_output_dir / "d_icd_diagnoses.csv",
            diagnoses_icd=self._sql_output_dir / "diagnoses_icd.csv",
            icustay_detail=self._sql_output_dir / "icustay_detail.csv"
        )



class PreprocessSettingsOld:
    def __init__(
            self,
            project_root: Path,
            min_age: int = 18,
            min_los_hospital: int = 1,
            min_los_icu: int = 1,
            num_diagnoses: int = 25,
            sql_result_relpath: Path = Path("data/mimiciii_query_results"),
            prefilter_output_relpath: Path = Path("/data/prefilter_output"),
    ):
        self.project_root = project_root
        self.min_age = min_age
        self.min_los_hospital = min_los_hospital
        self.min_los_icu = min_los_icu
        self.num_diagnoses = num_diagnoses
        self.sql_result_dir = project_root / sql_result_relpath
        self.prefilter_output_dir = project_root / prefilter_output_relpath

    @property
    def icustay_detail_csv(self) -> Path:
        return self.mimiciii_query_results_dir / "icustay_detail.csv"

    @property
    def diagnoses_icd_csv(self) -> Path:
        return self.mimiciii_query_results_dir / "diagnoses_icd.csv"

    @property
    def d_icd_diagnoses_csv(self) -> Path:
        return self.mimiciii_query_results_dir / "d_icd_diagnoses.csv"

    @property
    def prefilter_settings(self) -> ip.PrefilterSettings:
        return ip.PrefilterSettings(
            min_age=self.min_age,
            min_los_hospital=self.min_los_hospital,
            min_los_icu=self.min_los_icu,
            output_dir=self.prefilter_output_dir
        )


DEFAULT_SETTINGS = PreprocessSettings(
    project_root=Path(__file__).parent.parent.parent
)
