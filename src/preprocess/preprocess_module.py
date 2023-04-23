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
    def import_object(self) -> PreprocessModuleInputObject:
        pass


class CSVFileForDataframe(PreprocessModuleInputSource):

    def __init__(self, csv_path: Path, label: str):
        super().__init__(label=label)
        self._csv_path = csv_path

    def import_object(self) -> PreprocessModuleInputObject:
        df = pd.read_csv(filepath_or_buffer=self._csv_path)
        df.columns = [name.lower() for name in df.columns]
        return PreprocessModuleInputObject(
            object=df, label=self.label
        )


class PickleFile(PreprocessModuleInputSource):

    def __init__(self, pickle_path: Path, label: str):
        super().__init__(label=label)
        self._pickle_path = pickle_path

    def import_object(self) -> PreprocessModuleInputObject:
        with self._pickle_path.open(mode="rb") as pickle_file:
            imported_pickle = pickle.load(pickle_file)
        return PreprocessModuleInputObject(
            object=imported_pickle, label=self.label
        )


class PreprocessModule(ABC):
    def __init__(self, raw_inputs: list[PreprocessModuleInputSource]):
        self._raw_inputs = raw_inputs

    @property
    @abstractmethod
    def raw_inputs(self) -> list[PreprocessModuleInputSource]:
        pass

    @abstractmethod
    def prefilter(self):

