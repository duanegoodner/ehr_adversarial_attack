from pathlib import Path
import interface_prefilter as ip


class Preprocessor:
    def __init__(
            self,
            prefilter: ip.AbstractPrefilter,
            icustay_detail_csv: Path,
            diagnoses_icd_csv: Path,
            d_icd_diagnoses_csv: Path

    ):
        self._prefilter = prefilter
        self._icustay_detail_csv = icustay_detail_csv
        self._diagnoses_icd_csv = diagnoses_icd_csv
        self._d_icd_diagnoses_csv = d_icd_diagnoses_csv

    def run_prefilter(self):
        return self._prefilter.process(
            icustay_detail_csv=self._icustay_detail_csv,
            diagnoses_icd_csv=self._diagnoses_icd_csv,
            d_icd_diagnoses_csv=self._d_icd_diagnoses_csv
        )
