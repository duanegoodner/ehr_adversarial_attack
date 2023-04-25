import copy
import pandas as pd
import pickle
from enum import Enum, auto
from pathlib import Path
from typing import Callable


class ResourceType(Enum):
    CSV = auto()
    PICKLE = auto()
    OBJECT = auto()


class ResourceImporter:
    _supported_file_types = {
        ".csv": ResourceType.CSV,
        ".pickle": ResourceType.PICKLE,
    }

    def _id_resource_type(self, resource: object) -> ResourceType:
        if isinstance(resource, (str, bytes, Path )):
            return self._validate_path(path=Path(resource))
        else:
            return ResourceType.OBJECT

    def _validate_path(self, path: Path) -> ResourceType:
        assert path.exists()
        file_extension = f".{path.name.split('.')[-1]}"
        file_type = self._supported_file_types.get(file_extension)
        assert file_type is not None
        return file_type

    @staticmethod
    def _import_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def _import_pickle(path: Path) -> object:
        with path.open(mode="rb") as p:
            result = pickle.load(p)
        return result

    @staticmethod
    def _import_py_object(item: object) -> object:
        return copy.deepcopy(item)

    @property
    def _import_dispatch(self) -> dict[ResourceType, Callable]:
        return {
            ResourceType.CSV: self._import_csv,
            ResourceType.PICKLE: self._import_pickle,
            ResourceType.OBJECT: self._import_py_object,

        }

    def import_resource(self, resource: object) -> object:
        resource_type = self._id_resource_type(resource=resource)
        return self._import_dispatch[resource_type](resource)


class ResourceExporter:

    _supported_file_types = [".pickle"]

    def export(self, resource: object, path: Path):
        self._validate_path(path=path)
        with path.open(mode="wb") as p:
            pickle.dump(obj=resource, file=p)

    def _validate_path(self, path: Path):
        assert f".{path.name.split('.')[-1]}" in self._supported_file_types




