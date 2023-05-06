import numpy as np
import dill
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import preprocess_module as pm
import preprocess_resource as pr
import prefilter_input_classes as pfin


class FullAdmissionListBuilder(pm.PreprocessModule):
    def __init__(
        self,
        settings=pfin.FullAdmissionListBuilderSettings(),
        incoming_resource_refs=pfin.FullAdmissionListBuilderResourceRefs,
    ):
        super().__init__(
            settings=settings,
            incoming_resource_refs=incoming_resource_refs,
            resource_container_constructor=pfin.FullAdmissionListBuilderResources,
        )

    # have this for better autocomplete in process() (do same in prefilter.py)
    def call_import_resources(self) -> pfin.FullAdmissionListBuilderResources:
        return self._import_resources()

    def process(self):
        data = self.call_import_resources()
        data.icustay_bg_lab_vital = data.icustay_bg_lab_vital.drop(
            [
                "dod",
                "los_hospital",
                "admission_age",
                "hospstay_seq",
                "icustay_seq",
                "first_hosp_stay",
                "los_icu",
                "first_icu_stay",
            ], axis=1
        )

        df_grouped_by_hadm = data.icustay_bg_lab_vital.groupby(["hadm_id"])
        list_of_group_dfs = [group[1] for group in df_grouped_by_hadm]
        full_admission_data_list = [
            pfin.FullAdmissionData(
                subject_id=np.unique(item.subject_id),
                hadm_id=np.unique(item.hadm_id),
                icustay_id=np.unique(item.icustay_id),
                admittime=np.unique(item.admittime),
                dischtime=np.unique(item.dischtime),
                hospital_expire_flag=np.unique(item.hospital_expire_flag),
                intime=np.unique(item.intime),
                outtime=np.unique(item.outtime),
                time_series=item[self.settings.time_series_cols],
            )
            for item in list_of_group_dfs
        ]

        output_path = self.settings.output_dir / "full_admission_list.pickle"
        dill.dump(full_admission_data_list, output_path.open(mode="wb"))

        return pr.ExportedPreprocessResource(
            path=output_path, data_type=".pickle"
        )


if __name__ == "__main__":
    full_admission_list_builder = FullAdmissionListBuilder()
    exported_resources = full_admission_list_builder()



