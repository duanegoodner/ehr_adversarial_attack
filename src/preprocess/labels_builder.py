import pandas as pd
import numpy as np
import dataframe_provider as dfp


class LabelsBuilder:
    def __init__(
        self,
        icustay_detail: pd.DataFrame,
        diagnoses_icd: pd.DataFrame,
        d_icd_diagnoses: pd.DataFrame,
        min_patient_age: int = 18,
        min_los_days: int = 1,
    ):
        self._icustay_detail = icustay_detail
        self._diagnoses_icd = diagnoses_icd
        self._d_icd_diagnoses = d_icd_diagnoses
        self._min_patient_age = min_patient_age
        self._min_los_days = min_los_days

    @classmethod
    def from_df_provider(
        cls,
        df_provider: dfp.DataFrameProvider,
        min_patient_age: int = 18,
        min_los_days: int = 1,
    ):
        return cls(
            icustay_detail=df_provider.import_query_result("icustay_detail"),
            diagnoses_icd=df_provider.import_query_result("diagnoses_icd"),
            d_icd_diagnoses=df_provider.import_query_result("d_icd_diagnoses"),
            min_patient_age=min_patient_age,
            min_los_days=min_los_days,
        )

    @property
    def filtered_icustay_detail(self):
        return self._icustay_detail[
            (self._icustay_detail.admission_age >= 18)
            & (self._icustay_detail.los_icu >= 1)
            ]

    @property
    def filtered_diagnoses_icd(self):
        return self._diagnoses_icd[
            self._diagnoses_icd["subject_id"].isin(
                self.filtered_icustay_detail["subject_id"]
            )
        ].dropna()

    @property
    def reported_d_icd_diagnoses(self):
        reported_codes = np.unique(self.filtered_diagnoses_icd["icd9_code"])
        return self._d_icd_diagnoses[
            self._d_icd_diagnoses["icd9_code"].isin(
                reported_codes
            )]


if __name__ == "__main__":
    labels_builder = LabelsBuilder.from_df_provider(
        dfp.DEFAULT_DATAFRAME_PROVIDER
    )
