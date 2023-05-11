import os
import subprocess
from pathlib import Path


class DockerPostgresDB:
    def __init__(
        self,
        project_name
    ):
        os.environ.setdefault(
            "LOCAL_PROJECT_ROOT", str(Path(__file__).parent.parent.parent)
        )
        self._docker_compose_path = (
            Path(__file__).parent / "docker-compose.yml"
        )

    def build_container(self):
        subprocess.run(
            ["docker-compose", "-f", str(self._docker_compose_path), "build"]
        )


postgres_db = DockerPostgresDB()
postgres_db.build_container()

