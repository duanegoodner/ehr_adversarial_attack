from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


class InitialFilterSettings(NamedTuple):
    min_age: int
    min_los_hosp: int
    min_los_icu: int


@dataclass
class PrefilterSettings:
    min_age: int
    min_los_hosp: int
    min_los_icu: int


class PreprocessSettings:
    def __init__(
        self,
        project_root: Path,
        query_result_format: str = ".csv",
        query_names: tuple[str] = (
            "admissions",
            "d_icd_diagnoses",
            "diagnoses_icd",
            "icustay_detail",
            "pivoted_bg",
            "pivoted_gc",
            "pivoted_lab",
            "pivoted_uo",
            "pivoted_vital",
        ),
        initial_filter_settings: InitialFilterSettings = InitialFilterSettings(
            min_age=18, min_los_hosp=1, min_los_icu=1
        ),
        num_diagnoses: int = 25
    ):
        self._project_root = project_root
        self.validate_data_dirs()
        self._query_result_format = query_result_format
        self._query_names = query_names
        self._initial_filter_settings = initial_filter_settings
        self._num_diagnoses = num_diagnoses

    @property
    def _project_data(self) -> Path:
        return self._project_root / "data"

    @property
    def query_names(self) -> tuple[str]:
        return self._query_names

    @property
    def query_result_format(self) -> str:
        return self._query_result_format

    @property
    def initial_filter_settings(self):
        return self._initial_filter_settings

    @property
    def mimiciii_query_results(self):
        return self._project_data / "mimiciii_query_results"

    @property
    def preprocessed_output(self):
        return self._project_data / "preprocessed_data"

    @property
    def num_diagnoses(self) -> int:
        return self._num_diagnoses

    def validate_data_dirs(self):
        assert self._project_root.is_dir()
        assert self._project_data.is_dir()
        assert self.mimiciii_query_results.is_dir()
        assert (
            self.preprocessed_output.is_dir()
            or not self.preprocessed_output.exists()
        )
        if not self.preprocessed_output.exists():
            self.preprocessed_output.mkdir()


DEFAULT_SETTINGS = PreprocessSettings(
    project_root=Path(__file__).parent.parent.parent
)
