import numpy as np
import pandas as pd
from dataclasses import dataclass
import preprocess_module as pm
import prefilter_input_classes as pfin


@dataclass
class FeatureFinalizerResources:
    processed_admission_list: list[pfin.FullAdmissionData]


class FeatureFinalizer(pm.PreprocessModule):
    def __init__(
        self,
        settings=pfin.FeatureFinalizerSettings(),
        incoming_resource_refs=pfin.FeatureFinalizerResourceRefs(),
    ):
        super().__init__(
            settings=settings, incoming_resource_refs=incoming_resource_refs
        )

    def _import_resources(self) -> FeatureFinalizerResources:
        imported_data = FeatureFinalizerResources(
            processed_admission_list=self.import_pickle_to_list(
                self.incoming_resource_refs.processed_admission_list
            )
        )
        return imported_data

    @staticmethod
    def _get_measurement_col_names(
        admission_list: list[pfin.FullAdmissionData],
    ) -> tuple:
        # confirm each sample's dataframe has same col names
        all_data_col_names = [
            item.time_series.columns for item in admission_list
        ]
        first_item_names = list(all_data_col_names[0])
        assert all(
            [(names == first_item_names).all() for names in all_data_col_names]
        )
        first_item_names.remove("charttime")

        # return as tuple (for fixed order)
        return tuple(first_item_names)

    def _get_feature_array(self, sample: pfin.FullAdmissionData) -> np.ndarray:
        observation_start_time = getattr(
            sample, self.settings.observation_window_start
        )
        observation_end_time = observation_start_time + pd.Timedelta(
            hours=self.settings.observation_window_in_hours
        )

        data_in_observation_window = sample.time_series[
            (sample.time_series["charttime"] >= observation_start_time[0])
            & (sample.time_series["charttime"] <= observation_end_time[0])
        ]
        # data_in_observation_window.drop(["charttime"], axis=1, inplace=True)

        return data_in_observation_window.loc[
            :, ~data_in_observation_window.columns.isin(["charttime"])
        ].values

    def process(self):
        assert self.settings.output_dir.exists()
        data = self._import_resources()
        measurement_col_names = self._get_measurement_col_names(
            data.processed_admission_list
        )

        measurement_data_list = []
        in_hospital_mortality_list = []

        for entry in data.processed_admission_list:
            measurement_data_list.append(self._get_feature_array(sample=entry))
            in_hospital_mortality_list.append(
                entry.hospital_expire_flag.item()
            )

        if self.settings.require_exact_num_hours:
            indices_with_exact_num_hours = [
                i
                for i in range(len(measurement_data_list))
                if measurement_data_list[i].shape[0]
                == self.settings.observation_window_in_hours
            ]

            measurement_data_list = [
                measurement_data_list[i] for i in indices_with_exact_num_hours
            ]
            in_hospital_mortality_list = [
                in_hospital_mortality_list[i]
                for i in indices_with_exact_num_hours
            ]

        # in_hospital_mortality_array = np.array(in_hospital_mortality)

        self.export_resource(
            key="measurement_col_names",
            resource=measurement_col_names,
            path=self.settings.output_dir / "measurement_col_names.pickle",
        )

        self.export_resource(
            key="measurement_data",
            resource=measurement_data_list,
            path=self.settings.output_dir / "measurement_data_list.pickle",
        )

        self.export_resource(
            key="in_hospital_mortality",
            resource=in_hospital_mortality_list,
            path=self.settings.output_dir
            / "in_hospital_mortality_list.pickle",
        )


if __name__ == "__main__":
    feature_finalizer = FeatureFinalizer()
    exported_resources = feature_finalizer()
