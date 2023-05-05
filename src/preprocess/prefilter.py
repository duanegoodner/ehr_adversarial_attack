import pandas as pd
import preprocess_module as pm
import prefilter_input_classes as pfin
from preprocess_settings import PREPROC_SETTINGS


class Prefilter(pm.PreprocessModule):
    def __init__(
        self,
        settings: pfin.PrefilterSettings,
        incoming_resource_refs: pfin.PrefilterResourceRefs,
    ):
        super().__init__(
            settings=settings,
            incoming_resource_refs=incoming_resource_refs,
            resource_container_constructor=pfin.PrefilterResources,
        )

    @staticmethod
    def _apply_standard_df_formatting(
        imported_resources: pfin.PrefilterResources,
    ):
        for resource in imported_resources.__dict__.values():
            if isinstance(resource, pd.DataFrame):
                resource.columns = [item.lower() for item in resource.columns]

    def _filter_icustay(self, df: pd.DataFrame) -> pd.DataFrame:
        df["admittime"] = pd.to_datetime(df["admittime"])
        df["dischtime"] = pd.to_datetime(df["dischtime"])
        df["intime"] = pd.to_datetime(df["intime"])
        df["outtime"] = pd.to_datetime(df["outtime"])

        df = df[
            (df["admission_age"] >= self._settings.min_age)
            & (df["los_hospital"] >= self._settings.min_los_hospital)
            & (df["los_icu"] >= self._settings.min_los_icu)
        ]

        df = df.drop(["ethnicity", "ethnicity_grouped", "gender"], axis=1)

        return df

    @staticmethod
    def _filter_diagnoses_icd(
        diagnoses_icd: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        filtered_diagnoses_icd = diagnoses_icd[
            diagnoses_icd["subject_id"].isin(icustay["subject_id"])
        ].dropna()
        filtered_diagnoses_icd["seq_num"] = filtered_diagnoses_icd[
            "seq_num"
        ].astype("int64")
        return filtered_diagnoses_icd

    @staticmethod
    def _filter_measurement_df(
        df: pd.DataFrame,
        identifier_cols: list[str],
        measurements_of_interest: list[str],
    ):
        df = df[identifier_cols + measurements_of_interest]
        df = df.dropna(subset=measurements_of_interest, how="all")
        return df

    def _filter_bg(
        self, bg: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        bg["charttime"] = pd.to_datetime(bg["charttime"])
        bg["icustay_id"] = (
            bg["icustay_id"].fillna(0).astype("int64")
        )
        bg = bg[
            bg["hadm_id"].isin(icustay["hadm_id"])
        ]
        bg = self._filter_measurement_df(
            df=bg,
            identifier_cols=["icustay_id", "hadm_id", "charttime"],
            measurements_of_interest=[
                "potassium",
                "calcium",
                "ph",
                "pco2",
                "lactate",
            ],
        )

        return bg

    def _filter_lab(
        self, lab: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        lab["icustay_id"] = (
            lab["icustay_id"].fillna(0).astype("int64")
        )
        lab["hadm_id"] = (
            lab["hadm_id"].fillna(0).astype("int64")
        )
        lab["charttime"] = pd.to_datetime(lab["charttime"])
        lab = lab[
            lab["hadm_id"].isin(icustay["hadm_id"])
        ]
        lab = self._filter_measurement_df(
            df=lab,
            identifier_cols=[
                "icustay_id",
                "hadm_id",
                "subject_id",
                "charttime",
            ],
            measurements_of_interest=[
                "albumin",
                "bun",
                "creatinine",
                "sodium",
                "bicarbonate",
                "glucose",
                "inr",
            ],
        )

        return lab

    def _filter_vital(
        self, vital: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        vital["charttime"] = pd.to_datetime(vital["charttime"])
        vital = vital[
            vital["icustay_id"].isin(icustay["icustay_id"])
        ]

        vital = self._filter_measurement_df(
            df=vital,
            identifier_cols=["icustay_id", "charttime"],
            measurements_of_interest=[
                "heartrate",
                "sysbp",
                "diasbp",
                "tempc",
                "resprate",
                "spo2",
                "glucose",
            ],
        )

        return vital

    @staticmethod
    def _filter_admissions(
        admissions: pd.DataFrame, icustay: pd.DataFrame
    ) -> pd.DataFrame:
        admissions["admittime"] = pd.to_datetime(admissions["admittime"])
        admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
        admissions = admissions[
            admissions["hadm_id"].isin(icustay["hadm_id"])
        ]
        admissions = admissions[
            [
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "deathtime",
                "admission_type",
                "hospital_expire_flag",
                "has_chartevents_data",
            ]
        ]
        return admissions

    # use for better autocomplete of imported_resources attrs in self.process()
    def _call_import_resources(self) -> pfin.PrefilterResources:
        return self._import_resources()

    def process(self):
        imported_resources = self._call_import_resources()
        self._apply_standard_df_formatting(
            imported_resources=imported_resources
        )
        imported_resources.icustay = self._filter_icustay(
            df=imported_resources.icustay
        )
        imported_resources.diagnoses_icd = self._filter_diagnoses_icd(
            diagnoses_icd=imported_resources.diagnoses_icd,
            icustay=imported_resources.icustay,
        )
        imported_resources.bg = self._filter_bg(
            bg=imported_resources.bg,
            icustay=imported_resources.icustay,
        )
        imported_resources.lab = self._filter_lab(
            lab=imported_resources.lab,
            icustay=imported_resources.icustay,
        )
        imported_resources.vital = self._filter_vital(
            vital=imported_resources.vital,
            icustay=imported_resources.icustay,
        )
        imported_resources.admissions = self._filter_admissions(
            admissions=imported_resources.admissions,
            icustay=imported_resources.icustay,
        )

        for key, val in imported_resources.__dict__.items():
            self._export_resource(
                key=key,
                resource=val,
                path=self._settings.output_dir / f"{key}.pickle",
            )


if __name__ == "__main__":
    prefilter = Prefilter(
        settings=PREPROC_SETTINGS.prefilter_settings,
        incoming_resource_refs=PREPROC_SETTINGS.prefilter_resource_refs,
    )
    exported_resources = prefilter()
