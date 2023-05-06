import numpy as np
import pandas as pd
import scipy.stats.mstats as mstats
import time
import preprocess_module as pm
import prefilter_input_classes as pfin
from dataclasses import dataclass
from preprocess_settings import PREPROC_SETTINGS


@dataclass
class FeatureBuilderOutput:
    id_bg: pd.DataFrame
    id_lab: pd.DataFrame
    id_vital: pd.DataFrame


# TODO return some methods to private after done troubleshooting
class FeatureBuilder(pm.PreprocessModule):
    def __init__(
        self,
        settings: pfin.FeatureBuilderSettings,
        incoming_resource_refs: pfin.PrefilterResourceRefs,
    ):
        super().__init__(
            settings=settings,
            incoming_resource_refs=incoming_resource_refs,
            resource_container_constructor=pfin.FeatureBuilderResources,
        )

    # have this for better autocomplete in process() (do same in prefilter.py)
    def call_import_resources(self) -> pfin.FeatureBuilderResources:
        return self._import_resources()

    @staticmethod
    def create_id_bg(bg: pd.DataFrame, icustay: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(
            left=icustay,
            right=bg,
            on=["hadm_id"],
            how="right",
            suffixes=("_icu", "_bg"),
        )[
            [
                "subject_id",
                "hadm_id",
                "icustay_id_icu",
                "icustay_id_bg",
                "charttime",
                "potassium",
                "calcium",
                "ph",
                "pco2",
                "lactate",
            ]
        ].rename(
            columns={"icustay_id_icu": "icustay_id"}
        )

    @staticmethod
    def create_id_lab(
        lab: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        return (
            pd.merge(
                left=icustay,
                right=lab,
                on=["hadm_id"],
                how="right",
                suffixes=("_icu", "_lab"),
            )
            .rename(columns={"subject_id_icu": "subject_id"})[
                [
                    "subject_id",
                    "hadm_id",
                    "icustay_id_icu",
                    "icustay_id_lab",
                    "charttime",
                    "albumin",
                    "bun",
                    "creatinine",
                    "sodium",
                    "bicarbonate",
                    "glucose",
                    "inr",
                ]
            ]
            .rename(columns={"icustay_id_icu": "icustay_id"})
        )

    @staticmethod
    def create_id_vital(
        vital: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.merge(left=icustay, right=vital, on=["icustay_id"])[
            [
                "subject_id",
                "hadm_id",
                "icustay_id",
                "charttime",
                "heartrate",
                "sysbp",
                "diasbp",
                "tempc",
                "resprate",
                "spo2",
                "glucose",
            ]
        ]

    @staticmethod
    def merge_measurement_sources(
        id_bg: pd.DataFrame,
        id_lab: pd.DataFrame,
        id_vital: pd.DataFrame,
    ):
        id_bg_lab = pd.merge(
            left=id_bg,
            right=id_lab,
            on=["subject_id", "hadm_id", "icustay_id", "charttime"],
            how="outer",
        )
        id_bg_lab_vital = (
            pd.merge(
                left=id_bg_lab,
                right=id_vital,
                on=["subject_id", "hadm_id", "icustay_id", "charttime"],
                how="outer",
                suffixes=("_bglab", "_vital"),
            )
            .drop(["glucose_vital"], axis=1)
            .rename(columns={"glucose_bglab": "glucose"})
        )

        return id_bg_lab_vital

    @staticmethod
    def combine_icustay_info_with_measurements(
        id_bg_lab_vital: pd.DataFrame,
        icustay: pd.DataFrame,
    ) -> pd.DataFrame:
        info_id_bg_lab_vital = pd.merge(
            left=icustay,
            right=id_bg_lab_vital,
            on=["subject_id", "hadm_id", "icustay_id"],
            how="outer",
        ).drop(
            [
                "icustay_id_bg",
                "icustay_id_lab",
                "admission_age",
                "hospstay_seq",
                "first_hosp_stay",
                "icustay_seq",
                "first_icu_stay",
            ],
            axis=1,
        )

        return info_id_bg_lab_vital

    @staticmethod
    def winsorize(
        df: pd.DataFrame,
        cols: list[str],
        lower_cutoff: float,
        upper_cutoff: float,
    ):
        q_lower = df[cols].quantile(q=lower_cutoff)
        q_upper = df[cols].quantile(q=upper_cutoff)
        df[cols] = df[cols].clip(lower=q_lower, upper=q_upper, axis=1)

    # def develop(self) -> pd.DataFrame:
    #     data = self.call_import_resources()
    #     id_bg = self.create_id_bg(bg=data.bg, icustay=data.icustay)
    #     id_lab = self.create_id_lab(lab=data.lab, icustay=data.icustay)
    #     id_vital = self.create_id_vital(vital=data.vital, icustay=data.icustay)
    #
    #     icustay_bg_lab_vital = self.merge_measurement_sources(
    #         id_bg=id_bg, id_lab=id_lab, id_vital=id_vital
    #     )
    #
    #     self.winsorize(
    #         df=icustay_bg_lab_vital,
    #         cols=self._settings.all_measurement_cols,
    #         lower_cutoff=self._settings.winsorize_lower,
    #         upper_cutoff=self._settings.winsorize_upper,
    #     )
    #     return icustay_bg_lab_vital
    #
    def process(self):
        pass
        # data = self.call_import_resources()
        # id_bg = self.create_id_bg(bg=data.bg, icustay=data.icustay)
        # id_lab = self.create_id_lab(lab=data.lab, icustay=data.icustay)
        # id_vital = self.create_id_vital(vital=data.vital, icustay=data.icustay)


if __name__ == "__main__":
    feature_builder = FeatureBuilder(
        settings=PREPROC_SETTINGS.feature_builder_settings,
        incoming_resource_refs=PREPROC_SETTINGS.feature_builder_resource_refs,
    )

    my_data = feature_builder.call_import_resources()
    my_id_bg = feature_builder.create_id_bg(
        bg=my_data.bg, icustay=my_data.icustay
    )
    my_id_lab = feature_builder.create_id_lab(
        lab=my_data.lab, icustay=my_data.icustay
    )
    my_id_vital = feature_builder.create_id_vital(
        vital=my_data.vital, icustay=my_data.icustay
    )

    my_id_bg_lab_vital = feature_builder.merge_measurement_sources(
        id_bg=my_id_bg, id_lab=my_id_lab, id_vital=my_id_vital
    )

    my_icustay_bg_lab_vital = (
        feature_builder.combine_icustay_info_with_measurements(
            id_bg_lab_vital=my_id_bg_lab_vital, icustay=my_data.icustay
        )
    )

    feature_builder.winsorize(
        df=my_icustay_bg_lab_vital,
        cols=feature_builder._settings.all_measurement_cols,
        lower_cutoff=feature_builder._settings.winsorize_lower,
        upper_cutoff=feature_builder._settings.winsorize_upper,
    )

    groups = my_icustay_bg_lab_vital.groupby(["hadm_id"])
    groups_iter = iter(groups)
    grp_a = next(groups_iter)
    grp_b = next(groups_iter)
    grp_c = next(groups_iter)

    # result = feature_builder.develop()
