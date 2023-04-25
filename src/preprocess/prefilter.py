from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import preprocess_module as pm
import preprocess_resource as pr


@dataclass
class PrefilterResources:
    # admissions: pr.PreprocessResource
    d_icd_diagnoses: pr.IncomingPreprocessResource
    diagnoses_icd: pr.IncomingPreprocessResource
    icustay_detail: pr.IncomingPreprocessResource
    # pivoted_bg: pr.PreprocessResource
    # pivoted_gc: pr.PreprocessResource
    # pivoted_lab: pr.PreprocessResource
    # pivoted_uo: pr.PreprocessResource
    # pivoted_vital: pr.PreprocessResource


@dataclass
class ImportedPrefilterResources:
    d_icd_diagnoses: pd.DataFrame
    diagnoses_icd: pd.DataFrame
    icustay_detail: pd.DataFrame


@dataclass
class PrefilterSettings:
    output_dir: Path
    min_age: int = 18
    min_los_hospital: int = 1
    min_los_icu: int = 1


class Prefilter(pm.PreprocessModule):
    def __init__(
        self,
        settings: PrefilterSettings,
        incoming_resources: dict[str, pr.IncomingPreprocessResource],
    ):
        super().__init__(
            settings=settings, incoming_resources=incoming_resources
        )

    @staticmethod
    def _apply_standard_df_formatting(
        imported_resources: ImportedPrefilterResources,
    ):
        for resource in imported_resources.__dict__:
            if isinstance(resource, pd.DataFrame):
                resource.columns = [item.lower() for item in resource.columns]

    def _filter_icustay_detail(
        self, icustay_detail: pd.DataFrame
    ) -> pd.DataFrame:
        return icustay_detail[
            (icustay_detail.admission_age >= self._settings.min_age)
            & (icustay_detail.los_hospital >= self._settings.min_los_hospital)
            & (icustay_detail.los_icu >= self._settings.min_los_icu)
        ]

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

    def _process(self):
        imported_resources = ImportedPrefilterResources(
            **{
                key: resource.import_py_object()
                for key, resource in self._incoming_resources.items()
            }
        )

        self._apply_standard_df_formatting(
            imported_resources=imported_resources
        )
        imported_resources.icustay_detail = self._filter_icustay_detail(
            icustay_detail=imported_resources.icustay_detail
        )
        imported_resources.diagnoses_icd = self._filter_diagnoses_icd(
            diagnoses_icd=imported_resources.diagnoses_icd,
            icustay_detail=imported_resources.icustay_detail
        )
        
