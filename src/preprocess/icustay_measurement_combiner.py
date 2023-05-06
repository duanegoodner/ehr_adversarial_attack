import pandas as pd
import preprocess_module as pm
import prefilter_input_classes as pfin


class ICUStayMeasurementCombiner(pm.PreprocessModule):
    def __init__(
        self,
        settings=pfin.ICUStayMeasurementCombinerSettings(),
        incoming_resource_refs=pfin.ICUStayMeasurementCombinerResourceRefs(),
    ):
        super().__init__(
            settings=settings,
            incoming_resource_refs=incoming_resource_refs,
            resource_container_constructor=pfin.ICUStayMeasurementCombinerResources,
        )

    # have this for better autocomplete in process() (do same in prefilter.py)
    def call_import_resources(self) -> pfin.FeatureBuilderResources:
        return self._import_resources()

    def create_id_bg(
        self, bg: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
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
                "charttime",
            ]
            + self.settings.bg_data_cols
        ].rename(
            columns={"icustay_id_icu": "icustay_id"}
        )

    def create_id_lab(
        self, lab: pd.DataFrame, icustay: pd.DataFrame
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
                    "charttime",
                ]
                + self.settings.lab_data_cols
            ]
            .rename(columns={"icustay_id_icu": "icustay_id"})
        )

    # @staticmethod
    def create_id_vital(
        self, vital: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.merge(left=icustay, right=vital, on=["icustay_id"])[
            [
                "subject_id",
                "hadm_id",
                "icustay_id",
                "charttime",
            ]
            + self.settings.vital_data_cols
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
        )

        return info_id_bg_lab_vital

    def process(self):
        data = self.call_import_resources()
        id_bg = self.create_id_bg(bg=data.bg, icustay=data.icustay)
        id_lab = self.create_id_lab(lab=data.lab, icustay=data.icustay)
        id_vital = self.create_id_vital(vital=data.vital, icustay=data.icustay)
        id_bg_lab_vital = self.merge_measurement_sources(
            id_bg=id_bg, id_lab=id_lab, id_vital=id_vital
        )
        icustay_bg_lab_vital = self.combine_icustay_info_with_measurements(
            id_bg_lab_vital=id_bg_lab_vital, icustay=data.icustay
        )
        self._export_resource(
            key="icustay_bg_lab_vital",
            resource=icustay_bg_lab_vital,
            path=self.settings.output_dir / "icustay_bg_lab_vital.pickle",
        )


if __name__ == "__main__":
    icustay_measurement_combiner = ICUStayMeasurementCombiner()
    exported_resources = icustay_measurement_combiner()
