from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import preprocess_module as pm


@dataclass
class PrefilterResourceRefs:
    # admissions: pr.PreprocessResource
    d_icd_diagnoses: Path
    diagnoses_icd: Path
    icustay_detail: Path
    # pivoted_bg: pr.PreprocessResource
    # pivoted_gc: pr.PreprocessResource
    # pivoted_lab: pr.PreprocessResource
    # pivoted_uo: pr.PreprocessResource
    # pivoted_vital: pr.PreprocessResource


@dataclass
class PrefilterResources:
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
        incoming_resource_refs: PrefilterResourceRefs,
    ):
        super().__init__(
            settings=settings,
            incoming_resource_refs=incoming_resource_refs,
        )

    @staticmethod
    def _apply_standard_df_formatting(
        imported_resources: PrefilterResources,
    ):
        for resource in imported_resources.__dict__.values():
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

    def process(self):
        # Could make constructor a data member for more abstraction, but
        # then don't get as much IDE auto-complete help
        imported_resources = PrefilterResources(
            **{
                key: self._importer.import_resource(val)
                for key, val in self._incoming_resource_refs.__dict__.items()
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
            icustay_detail=imported_resources.icustay_detail,
        )

        for key, val in imported_resources.__dict__.items():
            self._export_resource(
                key=key,
                resource=val,
                path=self._settings.output_dir / f"{key}.pickle",
            )


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    sql_result_dir = data_dir / "mimiciii_query_results"
    prefilter_resources = PrefilterResourceRefs(
        d_icd_diagnoses=sql_result_dir / "d_icd_diagnoses.csv",
        diagnoses_icd=sql_result_dir / "diagnoses_icd.csv",
        icustay_detail=sql_result_dir / "icustay_detail.csv",
    )
    prefilter_settings = PrefilterSettings(
        output_dir=data_dir / "prefilter_output"
    )
    prefilter = Prefilter(
        settings=prefilter_settings, incoming_resource_refs=prefilter_resources
    )
    exported_resources = prefilter()
