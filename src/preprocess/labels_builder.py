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
            icustay_detail=df_provider.import_query_result(
                query_name="icustay_detail"
            ),
            diagnoses_icd=df_provider.import_query_result(
                query_name="diagnoses_icd"
            ),
            d_icd_diagnoses=df_provider.import_query_result(
                query_name="d_icd_diagnoses"
            ),
            min_patient_age=min_patient_age,
            min_los_days=min_los_days,
        )

    @property
    def filtered_icustay_detail(self) -> pd.DataFrame:
        return self._icustay_detail[
            (self._icustay_detail.admission_age >= 18)
            & (self._icustay_detail.los_icu >= 1)
        ]

    @property
    def filtered_diagnoses_icd(self) -> pd.DataFrame:
        filtered_df = self._diagnoses_icd[
            self._diagnoses_icd["subject_id"].isin(
                self.filtered_icustay_detail["subject_id"]
            )
        ].dropna()
        filtered_df["seq_num"] = filtered_df["seq_num"].astype("int64")
        return filtered_df

    @property
    def top_25_diagnoses(self):
        icd9_code_freq_tuple = np.unique(
            self.filtered_diagnoses_icd["icd9_code"], return_counts=True
        )
        icd9_code_freq = pd.DataFrame(
            list(zip(*icd9_code_freq_tuple)),
            columns=["icd9_code", "num_occurrences"],
        ).sort_values(by="num_occurrences", ascending=False)
        icd9_top_25 = icd9_code_freq.iloc[:25, :]
        return pd.merge(
            left=icd9_top_25,
            right=self._d_icd_diagnoses,
            on="icd9_code",
            how="inner",
        )[["icd9_code", "num_occurrences", "short_title"]]


if __name__ == "__main__":
    builder = LabelsBuilder.from_df_provider(dfp.DEFAULT_DATAFRAME_PROVIDER)
