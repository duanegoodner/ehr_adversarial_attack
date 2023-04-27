from pathlib import Path
import labels_builder_input_classes as lbin
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
            min_los_icu=1,
        )
        self._prefilter_resource_refs = pfin.PrefilterResourceRefs(
            admissions=self._sql_output_dir / "admissions.csv",
            d_icd_diagnoses=self._sql_output_dir / "d_icd_diagnoses.csv",
            diagnoses_icd=self._sql_output_dir / "diagnoses_icd.csv",
            icustay_detail=self._sql_output_dir / "icustay_detail.csv",
            pivoted_bg=self._sql_output_dir / "pivoted_bg.csv",
            pivoted_vital=self._sql_output_dir / "pivoted_vital.csv",
            pivoted_lab=self._sql_output_dir / "pivoted_lab.csv"
        )
        self._labels_builder_settings = lbin.LabelsBuilderSettings(
            output_dir=self._data_dir / "labels_builder_output",
            num_diagnoses=25,
        )
        self._labels_builder_resource_refs = lbin.LabelsBuilderResourceRefs(
            d_icd_diagnoses=self._data_dir
            / "prefilter_output"
            / "d_icd_diagnoses.pickle",
            diagnoses_icd=self._data_dir
            / "prefilter_output"
            / "diagnoses_icd.pickle",
        )

    @property
    def prefilter_settings(self) -> pfin.PrefilterSettings:
        return self._prefilter_settings

    @property
    def prefilter_resource_refs(self) -> pfin.PrefilterResourceRefs:
        return self._prefilter_resource_refs

    @property
    def labels_builder_settings(self) -> lbin.LabelsBuilderSettings:
        return self._labels_builder_settings

    @property
    def labels_builder_resource_refs(self) -> lbin.LabelsBuilderResourceRefs:
        return self._labels_builder_resource_refs


PREPROC_SETTINGS = PreprocessSettings(
    project_root=Path(__file__).parent.parent.parent
)
