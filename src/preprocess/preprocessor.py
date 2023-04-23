from pathlib import Path
import preprocess_module as pm


# TODO create module-specific data struct for each module input
class Preprocessor:
    def __init__(
            self,
            icustay_detail_csv: Path,
            diagnoses_icd_csv: Path,
            top_n_diagnoses: int,
            min_los_hosp: int,
            min_los_icu: int
    ):
        self._icustay_detail_csv = icustay_detail_csv
        self._diagnoses_icd_csv = diagnoses_icd_csv
        self._top_n_diagnoses = top_n_diagnoses
        self._min_los_hosp = min_los_hosp
        self._min_los_icu = min_los_icu

    def calc_top_diagnoses_freq(self):
        icustay_detail = pm.CSVFileForDataframe(
            csv_path=self._icustay_detail_csv,
            label="icustay_detail"
        )
        diagnoses_icd = pm.CSVFileForDataframe(
            csv_path=self._diagnoses_icd_csv,
            label="diagnoses_icd"
        )

