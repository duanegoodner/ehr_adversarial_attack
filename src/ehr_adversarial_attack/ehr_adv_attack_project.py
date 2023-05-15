from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataResource:
    path: Path
    py_object_type: str


class MimicIIIDatabaseInterface(ABC):
    @abstractmethod
    def run_sql_queries(
        self, sql_queries: list[Path]
    ) -> dict[str, DataResource]:
        pass


class DataPreprocessor(ABC):
    @abstractmethod
    def preprocess(
        self, sql_query_results: dict[str, DataResource]
    ) -> dict[str, DataResource]:
        pass


class ModelTrainer(ABC):



class EHRAdvAttackProject:
    def __init__(
        self,
        db_interface: MimicIIIDatabaseInterface,
        preprocessor: DataPreprocessor,
    ):
        self._db_interface = db_interface
        self._preprocessor = preprocessor
