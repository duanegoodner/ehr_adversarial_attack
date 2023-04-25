import copy
import pandas as pd
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class PreprocessModuleInputSource(ABC):
    @abstractmethod
    def import_object(self) -> object:
        pass


class CSVFileForDataframe(PreprocessModuleInputSource):
    def __init__(self, csv_path: Path):
        self._csv_path = csv_path

    def import_object(self) -> pd.DataFrame:
        df = pd.read_csv(filepath_or_buffer=self._csv_path)
        df.columns = [name.lower() for name in df.columns]
        return df


class PickleFile(PreprocessModuleInputSource):
    def __init__(self, pickle_path: Path):
        self._pickle_path = pickle_path

    def import_object(self) -> object:
        with self._pickle_path.open(mode="rb") as pickle_file:
            imported_pickle = pickle.load(pickle_file)
        return imported_pickle


class ExistingObject(PreprocessModuleInputSource):
    def __init__(self, existing_object: object):
        self._existing_object = existing_object

    def import_object(self) -> object:
        return copy.deepcopy(self._existing_object)


@dataclass
class PreprocessModuleOutputItem:
    name: str
    reference: Path | object