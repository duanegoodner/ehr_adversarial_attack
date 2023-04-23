from pathlib import Path
import preprocess_module as pm


class Prefilter(pm.PreprocessModule):
    def __init__(
            self,
            raw_inputs: list[pm.PreprocessModuleInputSource],
            min_age: int,
            min_los_hosp: int,
            min_los_icu: int,
    ):
        super().__init__(raw_inputs=raw_inputs)
        self._min_age = min_age
        self._min_los_hosp = min_los_hosp
        self._min_los_icu = min_los_icu


if __name__ == "__main__":
    icustay_detail_csv = (
            Path(__file__).parent.parent.parent
            / "data"
            / "mimiciii_query_results"
            / "icustay_detail.csv"
    )
    diagnoses_icd_csv = (
            Path(__file__).parent.parent.parent
            / "data"
            / "mimiciii_query_results"
            / "diagnoses_icd.csv"
    )
    icustay_detail = pm.CSVFileForDataframe(
        csv_path=icustay_detail_csv,
        label="icustay_detail"
    )
    diagnoses_icd = pm.CSVFileForDataframe(
        csv_path=diagnoses_icd_csv,
        label="diagnoses_icd"
    )

    prefilter = Prefilter(
        raw_inputs=[icustay_detail, diagnoses_icd],
        min_age=18,
        min_los_hosp=1,
        min_los_icu=1
    )
