from pathlib import Path
import labels_builder_input_classes as lbin
import prefilter_input_classes as pfin


class PreprocessSettings:
    def __init__(self, project_root: Path):
        self._project_root = project_root
        self._data_dir = project_root / "data"
        self._sql_output_dir = self._data_dir / "mimiciii_query_results"
        self._prefilter_output_dir = self._data_dir / "prefilter_output"
        self._prefilter_settings = pfin.PrefilterSettings(
            output_dir=self._prefilter_output_dir,
            min_age=18,
            min_los_hospital=1,
            min_los_icu=1,
        )
        self._prefilter_resource_refs = pfin.PrefilterResourceRefs(
            icustay=self._sql_output_dir / "icustay_detail.csv",
            bg=self._sql_output_dir / "pivoted_bg.csv",
            vital=self._sql_output_dir / "pivoted_vital.csv",
            lab=self._sql_output_dir / "pivoted_lab.csv",
        )
        self._feature_builder_settings = pfin.FeatureBuilderSettings(
            output_dir=self._data_dir / "feature_builder_output",
            winsorize_upper=0.95,
            winsorize_lower=0.05,
            bg_data_cols=["potassium", "calcium", "ph", "pco2", "lactate"],
            lab_data_cols=[
                "albumin",
                "bun",
                "creatinine",
                "sodium",
                "bicarbonate",
                "glucose",
                "inr",
            ],
            vital_data_cols=[
                "heartrate",
                "sysbp",
                "diasbp",
                "tempc",
                "resprate",
                "spo2",
            ],
        )
        self._feature_builder_resource_refs = pfin.FeatureBuilderResourceRefs(
            icustay=self._prefilter_output_dir / "icustay.pickle",
            bg=self._prefilter_output_dir / "bg.pickle",
            lab=self._prefilter_output_dir / "lab.pickle",
            vital=self._prefilter_output_dir / "vital.pickle",
        )

    @property
    def prefilter_settings(self) -> pfin.PrefilterSettings:
        return self._prefilter_settings

    @property
    def prefilter_resource_refs(self) -> pfin.PrefilterResourceRefs:
        return self._prefilter_resource_refs

    @property
    def feature_builder_settings(self) -> pfin.FeatureBuilderSettings:
        return self._feature_builder_settings

    @property
    def feature_builder_resource_refs(self) -> pfin.FeatureBuilderResourceRefs:
        return self._feature_builder_resource_refs


PREPROC_SETTINGS = PreprocessSettings(
    project_root=Path(__file__).parent.parent.parent
)
