from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import interface_prefilter as ip
from data_io import DataExporter


class Prefilter(ip.AbstractPrefilter):
    def __init__(
        self,
        settings: ip.PrefilterSettings,
    ):
        self._min_age = settings.min_age
        self._min_los_hospital = settings.min_los_hospital
        self._min_los_icu = settings.min_los_icu
        self._output_dir = settings.output_dir

    @staticmethod
    def set_standard_df_formatting(df: pd.DataFrame):
        df.columns = [label.lower() for label in df.columns]

    def filter_icustay_detail(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            (df.admission_age >= self._min_age)
            & (df.los_hospital >= self._min_los_hospital)
            & (df.los_icu >= self._min_los_icu)
        ]

    @staticmethod
    def filter_diagnoses_icd(
        diagnoses_icd: pd.DataFrame, icustay_detail: pd.DataFrame
    ) -> pd.DataFrame:
        filtered_diagnoses_icd = diagnoses_icd[
            diagnoses_icd["subject_id"].isin(icustay_detail["subject_id"])
        ].dropna()
        filtered_diagnoses_icd["seq_num"] = filtered_diagnoses_icd[
            "seq_num"
        ].astype("int64")
        return filtered_diagnoses_icd

    def write_output(
        self,
            icustay_detail_df: pd.DataFrame,
            diagnoses_icd_df: pd.DataFrame,
            d_icd_diagnoses_df: pd.DataFrame
    ) -> ip.PrefilterOutputs:
        data_exporter = DataExporter(output_dir=self._output_dir)

        icustay_detail_path = data_exporter.export_pickle(
            icustay_detail_df, "icustay_detail"
        )
        diagnoses_icd_path = data_exporter.export_pickle(
            diagnoses_icd_df, "diagnoses_icd"
        )
        d_icd_diagnoses_path = data_exporter.export_pickle(
            d_icd_diagnoses_df, "d_diagnoses_icd"
        )

        return ip.PrefilterOutputs(
            filtered_icustay_detail_pickle=icustay_detail_path,
            filtered_diagnoses_icd_pickle=diagnoses_icd_path,
            filtered_d_diagnoses_icd_pickle=d_icd_diagnoses_path
        )

    def process(
        self,
        icustay_detail_csv: Path,
        diagnoses_icd_csv: Path,
        d_icd_diagnoses_csv: Path
    ) -> ip.PrefilterOutputs:
        icustay_detail = pd.read_csv(icustay_detail_csv)
        diagnoses_icd = pd.read_csv(diagnoses_icd_csv)
        d_icd_diagnoses = pd.read_csv(d_icd_diagnoses_csv)

        self.set_standard_df_formatting(icustay_detail)
        self.set_standard_df_formatting(diagnoses_icd)
        self.set_standard_df_formatting(d_icd_diagnoses)
        icustay_detail = self.filter_icustay_detail(df=icustay_detail)
        diagnoses_icd = self.filter_diagnoses_icd(
            diagnoses_icd=diagnoses_icd,
            icustay_detail=icustay_detail,
        )

        return self.write_output(
            icustay_detail_df=icustay_detail,
            diagnoses_icd_df=diagnoses_icd,
            d_icd_diagnoses_df=d_icd_diagnoses
        )
