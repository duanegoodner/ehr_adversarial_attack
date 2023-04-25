from abc import ABC, abstractmethod
from dataclasses import dataclass
import preprocess_resource as pr


class PreprocessModule(ABC):
    def __init__(
        self,
        settings: dataclass,
        incoming_resources: dict[str, pr.IncomingPreprocessResource],
        exported_resources: dict[str, pr.ExportedPreprocessResource] = None,
    ):
        self._settings = settings
        self._incoming_resources = incoming_resources
        if exported_resources is None:
            exported_resources = {}
        self._exported_resources = exported_resources

    def __call__(
        self, *args, **kwargs
    ) -> dict[str, pr.ExportedPreprocessResource]:
        self._process()
        return self._exported_resources

    def _export_resource(
        self, key: str, resource: pr.OutgoingPreprocessResource
    ):
        exported_resource = resource.export()
        self._exported_resources[key] = exported_resource

    @abstractmethod
    def process(self):
        pass
