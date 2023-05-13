import os
import subprocess
import time
from pathlib import Path


class DockerPostgresDB:
    def __init__(
        self
    ):
        os.environ.setdefault(
            "LOCAL_PROJECT_ROOT", str(Path(__file__).parent.parent.parent)
        )
        local_db_owner =
        self._docker_compose_path = (
            Path(__file__).parent / "docker-compose.yml"
        )

    def build_image(self):
        subprocess.run(
            ["docker-compose", "-f", str(self._docker_compose_path), "build"]
        )

    def run_container(self):
        subprocess.run(
            ["docker-compose", "-f", str(self._docker_compose_path), "up", "-d"]
        )

    def stop_container(self):
        subprocess.run(
            ["docker-compose", "-f", str(self._docker_compose_path), "down"]
        )



postgres_db = DockerPostgresDB()
postgres_db.build_image()
postgres_db.run_container()
time.sleep(10)
postgres_db.stop_container()

