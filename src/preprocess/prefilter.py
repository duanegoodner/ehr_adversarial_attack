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
            resource_container_constructor=pfin.PrefilterResources,
        )

    @staticmethod
    def _apply_standard_df_formatting(
        imported_resources: pfin.PrefilterResources,
    ):
        for resource in imported_resources.__dict__.values():
            if isinstance(resource, pd.DataFrame):
                resource.columns = [item.lower() for item in resource.columns]

    def _filter_icustay_detail(self, df: pd.DataFrame) -> pd.DataFrame:
        df["admittime"] = pd.to_datetime(df["admittime"])
        df["dischtime"] = pd.to_datetime(df["dischtime"])
        df["intime"] = pd.to_datetime(df["intime"])
        df["outtime"] = pd.to_datetime(df["outtime"])

        df = df[
            (df["admission_age"] >= self._settings.min_age)
            & (df["los_hospital"] >= self._settings.min_los_hospital)
            & (df["los_icu"] >= self._settings.min_los_icu)
        ]
        return df

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

    @staticmethod
    def _filter_pivoted_bg(
        pivoted_bg: pd.DataFrame, icustay_detail: pd.DataFrame
    ) -> pd.DataFrame:
        pivoted_bg["charttime"] = pd.to_datetime(pivoted_bg["charttime"])
        pivoted_bg["icustay_id"] = (
            pivoted_bg["icustay_id"].fillna(0).astype("int64")
        )
        pivoted_bg = pivoted_bg[
            pivoted_bg["hadm_id"].isin(icustay_detail["hadm_id"])
        ]

        return pivoted_bg

    @staticmethod
    def _filter_pivoted_lab(
        pivoted_lab: pd.DataFrame, icustay_detail: pd.DataFrame
    ) -> pd.DataFrame:
        pivoted_lab["icustay_id"] = pivoted_lab["icustay_id"].fillna(0).astype(
            "int64"
        )
        pivoted_lab["hadm_id"] = pivoted_lab["hadm_id"].fillna(0).astype("int64")
        pivoted_lab["charttime"] = pd.to_datetime(pivoted_lab["charttime"])
        pivoted_lab = pivoted_lab[
            pivoted_lab["hadm_id"].isin(icustay_detail["hadm_id"])
        ]
        return pivoted_lab

    @staticmethod
    def _filter_pivoted_vital(
        pivoted_vital: pd.DataFrame, icustay_detail: pd.DataFrame
    ) -> pd.DataFrame:
        pivoted_vital["charttime"] = pd.to_datetime(pivoted_vital["charttime"])
        pivoted_vital = pivoted_vital[
            pivoted_vital["icustay_id"].isin(icustay_detail["icustay_id"])
        ]
        return pivoted_vital

    @staticmethod
    def _filter_admissions(
        admissions: pd.DataFrame, icustay_detail: pd.DataFrame
    ) -> pd.DataFrame:
        admissions["admittime"] = pd.to_datetime(admissions["admittime"])
        admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
        admissions = admissions[
            admissions["hadm_id"].isin(icustay_detail["hadm_id"])
        ]
        return admissions

    # use for better autocomplete of imported_resources attrs in self.process()
    def _call_import_resources(self) -> pfin.PrefilterResources:
        return self._import_resources()

    def process(self):
        imported_resources = self._call_import_resources()
        self._apply_standard_df_formatting(
            imported_resources=imported_resources
        )
        imported_resources.icustay_detail = self._filter_icustay_detail(
            df=imported_resources.icustay_detail
        )
        imported_resources.diagnoses_icd = self._filter_diagnoses_icd(
            diagnoses_icd=imported_resources.diagnoses_icd,
            icustay_detail=imported_resources.icustay_detail,
        )
        imported_resources.pivoted_bg = self._filter_pivoted_bg(
            pivoted_bg=imported_resources.pivoted_bg,
            icustay_detail=imported_resources.icustay_detail,
        )
        imported_resources.pivoted_lab = self._filter_pivoted_lab(
            pivoted_lab=imported_resources.pivoted_lab,
            icustay_detail=imported_resources.icustay_detail,
        )
        imported_resources.pivoted_vital = self._filter_pivoted_vital(
            pivoted_vital=imported_resources.pivoted_vital,
            icustay_detail=imported_resources.icustay_detail,

        )
        imported_resources.admissions = self._filter_admissions(
            admissions=imported_resources.admissions,
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
        incoming_resource_refs=PREPROC_SETTINGS.prefilter_resource_refs,
    )
    exported_resources = prefilter()
