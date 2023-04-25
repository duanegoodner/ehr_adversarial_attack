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


# TODO consider making ABC class for this to inherit from. ABC would have
#  TODO export method that would have proper format for
#   PreprocessModule._exported_resources
# TODO probably get rid of OutgoingPrefilterResources Container. Don't want to
#  be limited to exporting all at once.


@dataclass
class OutgoingPrefilterResources:
    d_icd_diagnoses: pr.OutgoingPreprocessResource
    diagnoses_icd: pr.OutgoingPreprocessResource
    icustay_detail: pr.OutgoingPreprocessResource


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

    def process(self):
        imported_resources = ImportedPrefilterResources(
            **{
                key: resource.import_py_object()
                for key, resource in self._incoming_resources.items()
            }
        )

        outgoing_d_icd_diagnoses = pr.OutgoingPreprocessResource(
            outgoing_object=imported_resources.diagnoses_icd,
            export_path=self._settings.output_dir
            / "filtered_d_icd_diagnoses.pickle",
        )

        self._export_resource(
            key="outgoing_d_icd_diagnoses", resource=outgoing_d_icd_diagnoses
        )

        self._apply_standard_df_formatting(
            imported_resources=imported_resources
        )
        imported_resources.icustay_detail = self._filter_icustay_detail(
            icustay_detail=imported_resources.icustay_detail
        )

        outgoing_icustay_detail = pr.OutgoingPreprocessResource(
            outgoing_object=imported_resources.icustay_detail,
            export_path=self._settings.output_dir
            / "filtered_icustay_detail.pickle",
        )

        self._export_resource(
            key="filtered_icustay_detail", resource=outgoing_icustay_detail
        )

        imported_resources.diagnoses_icd = self._filter_diagnoses_icd(
            diagnoses_icd=imported_resources.diagnoses_icd,
            icustay_detail=imported_resources.icustay_detail,
        )

        outgoing_diagnoses_icd = pr.OutgoingPreprocessResource(
            outgoing_object=imported_resources.diagnoses_icd,
            export_path=self._settings.output_dir
            / "filtered_diagnoses_icd.pickle",
        )

        self._export_resource(
            key="filtered_diagnoses_icd", resource=outgoing_diagnoses_icd
        )


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data"
    sql_result_dir = data_dir / "mimiciii_query_results"

    incoming_d_icd_diagnoses = pr.IncomingPreprocessResource(
        import_path=sql_result_dir / "d_icd_diagnoses.csv"
    )
    incoming_diagnoses_icd = pr.IncomingPreprocessResource(
        import_path=sql_result_dir / "diagnoses_icd.csv"
    )

    incoming_icustay_detail = pr.IncomingPreprocessResource(
        import_path=sql_result_dir / "icustay_detail.csv"
    )

    incoming_prefilter_resources = {
        "diagnoses_icd": incoming_diagnoses_icd,
        "d_icd_diagnoses": incoming_d_icd_diagnoses,
        "icustay_detail": incoming_icustay_detail,
    }

    prefilter_settings = PrefilterSettings(
        output_dir=data_dir / "prefilter_output"
    )

    prefilter = Prefilter(
        settings=prefilter_settings,
        incoming_resources=incoming_prefilter_resources,
    )

    prefilter.process()
