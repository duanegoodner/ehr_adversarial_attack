import numpy as np
import pandas as pd


class TopDiagnosesCalculator:
    def __init__(
        self,
        filtered_icustay_detail: pd.DataFrame,
        filtered_diagnoses_icd: pd.DataFrame,
        top_n: int,
    ):
        self._filtered_icustay_detail = filtered_icustay_detail
        self._filtered_diagnoses_icd = filtered_diagnoses_icd
        self._top_n = top_n

    def _code_frequencies(self) -> pd.DataFrame:
        icd9_code_freq_tuple = np.unique(
            self._filtered_diagnoses_icd["icd9_code"], return_counts=True
        )
        return pd.DataFrame(
            list(zip(*icd9_code_freq_tuple)),
            columns=["icd9_code", "num_occurrences"],
        )

    def labelled_top_n_codes(self) -> pd.DataFrame:
        return self._code_frequencies().sort_values(
            by="num_occurrences", ascending=False
        ).iloc[:self._top_n, :]

