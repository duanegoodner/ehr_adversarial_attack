import copy
import pandas as pd
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class IncomingPreprocessResource:
    _import_file_types = [".csv", ".pickle"]

    def __init__(
        self, import_path: Path = None, pre_existing_object: object = None
    ):
        assert (not import_path) ^ (not pre_existing_object)
        self._import_path = import_path
        self._pre_existing_object = pre_existing_object
        self._validate_import_path()

    def import_py_object(self) -> pd.DataFrame | object:
        if self._pre_existing_object:
            return copy.deepcopy(self._pre_existing_object)
        else:
            return self._import_dispatch[self._import_file_type]()

    def _validate_import_path(self):
        assert any(
            [
                self._import_path.name.endswith(ext)
                for ext in self._import_file_types
            ]
        )

    @property
    def _import_file_type(self) -> str:
        if self._import_path:
            return f".{self._import_path.name.split('.')[-1]}"

    @property
    def _import_dispatch(self) -> dict[str, Callable]:
        return {
            ".csv": self._import_from_csv,
            ".pickle": self._import_from_pickle,
        }

    def _import_from_csv(self):
        return pd.read_csv(self._import_path)

    def _import_from_pickle(self):
        with self._import_path.open(mode="rb") as p:
            py_object = pickle.load(p)
        return py_object


@dataclass
class ExportedPreprocessResource:
    path: Path
    data_type: str


class OutgoingPreprocessResource:
    _supported_export_file_type = ".pickle"

    def __init__(
        self,
        outgoing_object: object = None,
        export_path: Path = None,
    ):
        self._outgoing_object = outgoing_object
        self._export_path = export_path
        self._validate_export_path()

    @property
    def export_path(self) -> Path:
        return self._export_path

    def _validate_export_path(self):
        assert self._export_path.name.endswith(
            self._supported_export_file_type
        )

    def export(self) -> ExportedPreprocessResource:
        with self._export_path.open(mode="wb") as p:
            pickle.dump(self._outgoing_object, p)
        return ExportedPreprocessResource(
            path=self._export_path,
            data_type=type(self._outgoing_object).__name__,
        )
