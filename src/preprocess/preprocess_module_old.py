import copy
import pandas as pd
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PreprocessModuleInputObject:
    object: object
    label: str


class PreprocessModuleInputSource(ABC):
    def __init__(self, label: str):
        self.label = label

    @abstractmethod
    def import_object(self) -> object:
        pass


class CSVFileForDataframe(PreprocessModuleInputSource):

    def __init__(self, csv_path: Path, label: str):
        super().__init__(label=label)
        self._csv_path = csv_path

    def import_object(self) -> pd.DataFrame:
        df = pd.read_csv(filepath_or_buffer=self._csv_path)
        df.columns = [name.lower() for name in df.columns]
        return df


class PickleFile(PreprocessModuleInputSource):

    def __init__(self, pickle_path: Path, label: str):
        super().__init__(label=label)
        self._pickle_path = pickle_path

    def import_object(self) -> object:
        with self._pickle_path.open(mode="rb") as pickle_file:
            imported_pickle = pickle.load(pickle_file)
        return imported_pickle


class ExistingObject(PreprocessModuleInputSource):
    def __init__(self, existing_object: object, label: str):
        super().__init__(label=label)
        self._existing_object = existing_object

    def import_object(self) -> object:
        return copy.deepcopy(self._existing_object)


class PreprocessModule(ABC):
    def __init__(self, raw_inputs: list[PreprocessModuleInputSource]):
        self._input_objects = {
            item.label: item.import_object() for item in raw_inputs
        }
