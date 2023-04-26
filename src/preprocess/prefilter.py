import pandas as pd
import preprocess_module as pm
import prefilter_input_classes as pfin
from preprocess_settings import PREPROC_SETTINGS


class Prefilter(pm.PreprocessModule):
    def __init__(
        self,
        settings: pfin.PrefilterSettings,
        incoming_resource_refs: pfin.PrefilterResourceRefs,
    ):
        super().__init__(
            settings=settings,
            incoming_resource_refs=incoming_resource_refs,
            resource_container_constructor=pfin.PrefilterResources
        )

    @staticmethod
    def _apply_standard_df_formatting(
        imported_resources: pfin.PrefilterResources,
    ):
        for resource in imported_resources.__dict__.values():
            if isinstance(resource, pd.DataFrame):
                resource.columns = [item.lower() for item in resource.columns]

    def _filter_icustay_detail(
        self, icustay_detail: pd.DataFrame
    ) -> pd.DataFrame:
        return icustay_detail[
            (icustay_detail.admission_age >= self._settings.min_age)
            & (icustay_detail.los_hospital >= self._settings.min_los_hospital)
            & (icustay_detail.los_icu >= self._settings.min_los_icu)
        ]

    @staticmethod
    def _filter_diagnoses_icd(
        diagnoses_icd: pd.DataFrame, icustay_detail: pd.DataFrame
    ) -> pd.DataFrame:
        filtered_diagnoses_icd = diagnoses_icd[
            diagnoses_icd["subject_id"].isin(icustay_detail["subject_id"])
        ].dropna()
        filtered_diagnoses_icd["seq_num"] = filtered_diagnoses_icd[
            "seq_num"
        ].astype("int64")
        return filtered_diagnoses_icd

    # use for better autocomplete of imported_resources attrs in self.process()
    def _call_import_resources(self) -> pfin.PrefilterResources:
        return self._import_resources()

    def process(self):
        imported_resources = self._call_import_resources()
        self._apply_standard_df_formatting(
            imported_resources=imported_resources
        )
        imported_resources.icustay_detail = self._filter_icustay_detail(
            icustay_detail=imported_resources.icustay_detail
        )
        imported_resources.diagnoses_icd = self._filter_diagnoses_icd(
            diagnoses_icd=imported_resources.diagnoses_icd,
            icustay_detail=imported_resources.icustay_detail,
        )

        for key, val in imported_resources.__dict__.items():
            self._export_resource(
                key=key,
                resource=val,
                path=self._settings.output_dir / f"{key}.pickle",
            )


if __name__ == "__main__":
    prefilter = Prefilter(
        settings=PREPROC_SETTINGS.prefilter_settings,
        incoming_resource_refs=PREPROC_SETTINGS.prefilter_resource_refs
    )
    exported_resources = prefilter()
