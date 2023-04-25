import pandas as pd
import pickle
from pathlib import Path


class QueryResultNotFound(Exception):
    def __init__(self, path: Path):
        self.path = path

    def __str__(self):
        return f"{self.path} not found."


class QueryResultImporter:
    """
    Imports query results (saved as csvs in query_results_dir) as Dataframes
    """

    def __init__(self, query_results_dir: Path):
        self._query_results_dir = query_results_dir

    @property
    def query_results_dir(self) -> Path:
        return self._query_results_dir

    @property
    def available_query_results(self) -> list[str]:
        return [item.name for item in self._query_results_dir.glob("*.csv")]

    def import_query_result(self, filename: str) -> pd.DataFrame:
        if filename not in self.available_query_results:
            raise QueryResultNotFound(self._query_results_dir / filename)
        return pd.read_csv(self._query_results_dir / filename)


class DataExporter:
    def __init__(self, output_dir: Path):
        self._output_dir = output_dir

    def export_pickle(self, item: object, pickle_name: str) -> Path:
        pickle_file_path = self._output_dir / f"{pickle_name}.pickle"
        # assert not pickle_file_path.exists()
        with pickle_file_path.open(mode="wb") as p:
            pickle.dump(obj=item, file=p)
        return pickle_file_path


class PickleImporter:
    def __init__(self, pickle_path: Path):
        self._pickle_path = pickle_path

    # use different method depending on imported object
    # (just for type hinting benefit)
    def import_pickle_to_df(self) -> pd.DataFrame:
        with self._pickle_path.open(mode="rb") as p:
            return pickle.load(p)



class DataIO:
    def __init__(
            self,
            input_dir: Path,
            output_dir: Path,
    ):
        self._input_dir = input_dir
        self._output_dir = output_dir

    def import_query_result(self, query_name: str):
        df = pd.read_csv(
            self._input_dir / f"{query_name}.csv"
        )
        # df.columns = [item.lower() for item in list(df.columns)]
        return df

    def import_pickle(self, pickle_name: str):
        with (self._input_dir / pickle_name).open(mode="rb") as p:
            return pickle.load(p)

    def export_pickle(self, item: object, pickle_name: str):
        pickle_file_path = self._output_dir / f"{pickle_name}.pickle"
        assert not pickle_file_path.exists()
        with pickle_file_path.open(mode="wb") as p:
            pickle.dump(obj=item, file=p)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    mimiciii_query_results_dir = project_root / "data" / "mimiciii_query_results"
    project_pickle_dir = project_root / "data" / "pickles"
    data_io = DataIO(query_results_dir=mimiciii_query_results_dir, pickle_dir=project_pickle_dir)
    diagnoses_icd = data_io.import_query_result(
        query_name="diagnoses_icd"
    ).dropna()
    diagnoses_icd["seq_num"] = diagnoses_icd["seq_num"].astype("int64")
