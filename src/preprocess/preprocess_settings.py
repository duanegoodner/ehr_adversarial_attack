from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


@dataclass
class QueryBaseNames:
    admissions: str = "admissions"
    d_icd_diagnoses: str = "d_icd_diagnoses"
    diagnoses_icd: str = "diagnoses_icd"
    icustay_detail: str = "icustay_detail"
    pivoted_bg: str = "pivoted_bg"
    pivoted_gc: str = "pivoted_gc"
    pivoted_lab: str = "pivoted_lab"
    pivoted_uo: str = "pivoted_uo"
    pivoted_vital: str = "pivoted_vital"


class PreprocessSettings:
    def __init__(
        self,
        project_root: Path,
        query_result_format: str = ".csv",
        query_base_names: tuple[str] = (
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
    ):
        self.project_root = project_root
        self.validate_data_dirs()
        self.query_base_names = query_base_names
        self.query_result_format = query_result_format

    @property
    def project_data(self):
        return self.project_root / "data"

    @property
    def mimiciii_query_results(self):
        return self.project_data / "mimiciii_query_results"

    @property
    def preprocessed_output(self):
        return self.project_data / "preprocessed_data"

    def validate_data_dirs(self):
        assert self.project_root.is_dir()
        assert self.project_data.is_dir()
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
