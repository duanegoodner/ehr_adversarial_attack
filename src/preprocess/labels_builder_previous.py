from pathlib import Path
import inerface_labels_builder as ilb
import numpy as np
import pandas as pd
import data_io as io


class LabelsBuilder(ilb.AbstractLabelsBuilder):
    def __init__(self, num_diagnoses: int, output_dir: Path):
        self._num_diagnoses = num_diagnoses
        self._output_dir = output_dir

    # TODO write top_n_diagnoses method (copy from labels_builder_old.py)
    def calc_top_n_diagnoses(
        self, diagnoses_icd: pd.DataFrame, d_icd_diagnoses: pd.DataFrame
    ) -> pd.DataFrame:
        icd9_code_freq_tuple = np.unique(
            diagnoses_icd["icd9_code"], return_counts=True
        )
        icd9_code_freq = pd.DataFrame(
            list(zip(*icd9_code_freq_tuple)),
            columns=["icd9_code", "num_occurrences"],
        ).sort_values(by="num_occurrences", ascending=False)
        icd9_top_n = icd9_code_freq.iloc[: self._num_diagnoses, :]
        return pd.merge(
            left=icd9_top_n,
            right=d_icd_diagnoses,
            on="icd9_code",
            how="inner",
        )[["icd9_code", "num_occurrences", "short_title"]]

    def write_output(
        self, top_n_diagnoses: pd.DataFrame
    ) -> ilb.LabelsBuilderOutputs:
        data_exporter = io.DataExporter(output_dir=self._output_dir)
        top_n_diagnoses_path = data_exporter.export_pickle(
            item=top_n_diagnoses, pickle_name=f"top_n_diagnoses"
        )
        return ilb.LabelsBuilderOutputs(
            top_n_diagnoses_pickle=top_n_diagnoses_path
        )

    def process(
        self,
        filtered_diagnoses_icd_pickle: Path,
        filtered_d_icd_diagnoses_pickle: Path,
    ) -> ilb.LabelsBuilderOutputs:
        diagnoses_icd = io.PickleImporter(
            pickle_path=filtered_diagnoses_icd_pickle
        ).import_pickle_to_df()
        d_icd_diagnoses = io.PickleImporter(
            pickle_path=filtered_d_icd_diagnoses_pickle
        ).import_pickle_to_df()
        top_n_diagnoses = self.calc_top_n_diagnoses(
            diagnoses_icd=diagnoses_icd, d_icd_diagnoses=d_icd_diagnoses
        )
        return self.write_output(top_n_diagnoses=top_n_diagnoses)

