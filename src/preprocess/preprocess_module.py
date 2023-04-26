from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import preprocess_resource as pr
import resource_io as rio


class PreprocessModule(ABC):
    def __init__(
        self,
        settings: dataclass,
        incoming_resource_refs: dataclass,
        resource_container_constructor: Callable[..., dataclass],
        importer: rio.ResourceImporter = rio.ResourceImporter(),
        exporter: rio.ResourceExporter = rio.ResourceExporter(),
        exported_resources: dict[str, pr.ExportedPreprocessResource] = None,
    ):
        self._settings = settings
        self._importer = importer
        self._exporter = exporter
        self._resource_container_constructor = resource_container_constructor
        self._incoming_resource_refs = incoming_resource_refs
        if exported_resources is None:
            exported_resources = {}
        self._exported_resources = exported_resources

    # TODO Add ability to only import certain resource refs instead of all?
    def _import_resources(self) -> dataclass:
        resource_container = self._resource_container_constructor(
            **self._incoming_resource_refs.__dict__
        )
        assert sorted(self._incoming_resource_refs.__dict__.keys()) == sorted(
            resource_container.__dict__.keys()
        )
        for key, resource_ref in self._incoming_resource_refs.__dict__.items():
            setattr(
                resource_container,
                key,
                self._importer.import_resource(resource=resource_ref),
            )
        return resource_container

    def __call__(
        self, *args, **kwargs
    ) -> dict[str, pr.ExportedPreprocessResource]:
        self.process()
        return self._exported_resources

    def _export_resource(self, key: str, resource: object, path: Path):
        assert key not in self._exported_resources
        assert path not in [
            item.path for item in self._exported_resources.values()
        ]
        self._exporter.export(resource=resource, path=path)
        exported_resource = pr.ExportedPreprocessResource(
            path=path, data_type=type(resource).__name__
        )
        self._exported_resources[key] = exported_resource

    @abstractmethod
    def process(self):
        pass
