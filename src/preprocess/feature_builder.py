import pandas as pd
import preprocess_module as pm
import prefilter_input_classes as pfin
from preprocess_settings import PREPROC_SETTINGS


class FeatureBuilder(pm.PreprocessModule):
    def __init__(
        self,
        settings: pfin.FeatureBuilderSettings,
        incoming_resource_refs: pfin.PrefilterResourceRefs,
    ):
        super().__init__(
            settings=settings,
            incoming_resource_refs=incoming_resource_refs,
            resource_container_constructor=pfin.FeatureBuilderResources,
        )

    # have this for better autocomplete in process() (do same in prefilter.py)
    def _call_import_resources(self) -> pfin.FeatureBuilderResources:
        return self._import_resources()

    @staticmethod
    def _create_identified_bg(
        bg: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.merge(
            left=icustay,
            right=bg,
            on=["hadm_id"],
            how="right",
            suffixes=("_icu", "_bg"),
        )[
            [
                "subject_id",
                "hadm_id",
                "icustay_id_icu",
                "icustay_id_bg",
                "charttime",
                "potassium",
                "calcium",
                "ph",
                "pco2",
                "lactate",
            ]
        ]

    # def _create_identified_measurement_dfs(self):


    # def process(self):
    #     imported_resources = self._call_import_resources()
