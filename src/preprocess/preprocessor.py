import pandas as pd
from dataclasses import dataclass
from typing import Callable
import data_io as dfp
import preprocess_settings as ps
import top_diagnoses_calculator as tdc


class InitialFilter:
    def __init__(
        self,
        df_provider: dfp.DataFrameProvider,
        filter_settings: ps.InitialFilterSettings,
    ):
        self._filter_settings = filter_settings
        self._df_provider = df_provider

    def filtered_icustay_detail(self) -> pd.DataFrame:
        unfiltered_df = self._df_provider.import_query_result(
            query_name="icustay_detail"
        )
        return unfiltered_df[
            (unfiltered_df.admission_age >= self._filter_settings.min_age)
            & (unfiltered_df.los_icu >= self._filter_settings.min_los_icu)
        ]

    def filtered_diagnoses_icd(self) -> pd.DataFrame:
        unfiltered_df = self._df_provider.import_query_result(
            query_name="diagnoses_icd"
        )
        filtered_df = unfiltered_df[
            unfiltered_df["subject_id"].isin(
                self.filtered_icustay_detail()["subject_id"]
            )
        ].dropna()
        filtered_df["seq_num"] = filtered_df["seq_num"].astype("int64")
        return filtered_df

    @property
    def filter_dispatch(self) -> dict[str, Callable[..., pd.DataFrame]]:
        return {
            "icustay_detail": self.filtered_icustay_detail,
            "diagnoses_icd": self.filtered_diagnoses_icd,
        }

    def get_filtered_df(self, query_name: str) -> pd.DataFrame:
        if query_name in self.filter_dispatch:
            return self.filter_dispatch[query_name]()
        return self._df_provider.import_query_result(query_name=query_name)


class Preprocessor:
    def __init__(
        self,
        preprocess_settings: ps.PreprocessSettings = ps.DEFAULT_SETTINGS,
        df_provider: dfp.DataFrameProvider = dfp.DEFAULT_DATAFRAME_PROVIDER,
    ):
        self._preprocess_settings = preprocess_settings
        self._df_provider = df_provider
        self._initial_filter = InitialFilter(
            df_provider=df_provider,
            filter_settings=preprocess_settings.initial_filter_settings,
        )



    def run(self):
        filtered_icustay_detail = self._initial_filter.get_filtered_df(
            "icustay_detail"
        )
        filtered_diagnoses_icd = self._initial_filter.get_filtered_df(
            "diagnoses_icd"
        )
        filtered_d_icd_diagnoses = self._initial_filter.get_filtered_df(
            "d_icd_diagnoses"
        )
        top_diagnoses_codes = tdc.TopDiagnosesCalculator(
            filtered_icustay_detail=filtered_icustay_detail,
            filtered_diagnoses_icd=filtered_diagnoses_icd,
            top_n=self._preprocess_settings.num_diagnoses
        ).labelled_top_n_codes()

        return pd.merge(
            left=top_diagnoses_codes,
            right=filtered_d_icd_diagnoses,
            on="icd9_code",
            how="inner",
        )[["icd9_code", "num_occurrences", "short_title"]]


if __name__ == "__main__":
    p = Preprocessor()
    result = p.run()
