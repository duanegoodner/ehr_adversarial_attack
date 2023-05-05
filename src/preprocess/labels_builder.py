# Not needed, and will not work with current version of Preprocess
# (Preprocess does not provide diagnoses dataframes)

import numpy as np
import pandas as pd
import labels_builder_input_classes as lbin
import preprocess_module as pm
from preprocess_settings import PREPROC_SETTINGS


class LabelsBuilder(pm.PreprocessModule):
    def __init__(
        self,
        settings: lbin.LabelsBuilderSettings,
        incoming_resource_refs: lbin.LabelsBuilderResourceRefs,
    ):
        super().__init__(
            settings=settings,
            incoming_resource_refs=incoming_resource_refs,
            resource_container_constructor=lbin.LabelsBuilderResources,
        )

    def _call_import_resources(self) -> lbin.LabelsBuilderResources:
        return self._import_resources()

    def _calc_top_n_diagnoses(
        self, diagnoses_icd: pd.DataFrame, d_icd_diagnoses: pd.DataFrame
    ) -> pd.DataFrame:
        icd9_code_freq_tuple = np.unique(
            diagnoses_icd["icd9_code"], return_counts=True
        )
        icd9_code_freq = pd.DataFrame(
            list(zip(*icd9_code_freq_tuple)),
            columns=["icd9_code", "num_occurrences"],
        ).sort_values(by="num_occurrences", ascending=False)
        icd9_top_n = icd9_code_freq.iloc[: self._settings.num_diagnoses, :]
        return pd.merge(
            left=icd9_top_n,
            right=d_icd_diagnoses,
            on="icd9_code",
            how="inner",
        )[["icd9_code", "num_occurrences", "short_title"]]

    def process(self):
        imported_resources = self._call_import_resources()
        top_n_diagnoses = self._calc_top_n_diagnoses(
            diagnoses_icd=imported_resources.diagnoses_icd,
            d_icd_diagnoses=imported_resources.d_icd_diagnoses,
        )
        self._export_resource(
            key=f"top_{self._settings.num_diagnoses}_diagnoses",
            resource=top_n_diagnoses,
            path=self._settings.output_dir
            / f"top_{self._settings.num_diagnoses}_diagnoses.pickle",
        )


if __name__ == "__main__":
    labels_builder = LabelsBuilder(
        settings=PREPROC_SETTINGS.labels_builder_settings,
        incoming_resource_refs=PREPROC_SETTINGS.labels_builder_resource_refs,
    )
    exported_resources = labels_builder()

    re_imported_top_n = labels_builder._importer.import_resource(
        resource=PREPROC_SETTINGS.labels_builder_settings.output_dir
        / "top_25_diagnoses.pickle"
    )
