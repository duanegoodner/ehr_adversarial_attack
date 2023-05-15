import csv
import os
import psycopg2
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, TypedDict
from ehr_adv_attack_project import DataResource, MimiciiiDatabaseInterface


class DatabaseLoginInfo(TypedDict):
    host: str
    database: str
    user: str
    password: str


class MimiciiiDatabaseAccess(MimiciiiDatabaseInterface):
    def __init__(self, dotenv_path: Path, output_dir: Path):
        load_dotenv(dotenv_path=dotenv_path)
        self._connection = None
        self._dotenv_path = dotenv_path
        self._output_dir = output_dir


    def connect(self):
        load_dotenv(dotenv_path=self._dotenv_path)
        self._connection = psycopg2.connect(
            host=os.getenv("MIMICIII_DATABASE_HOST"),
            database=os.getenv("MIMICIII_DATABASE_NAME"),
            user=os.getenv("MIMICIII_DATABASE_USER"),
            password=os.getenv("MIMICIII_DATABASE_PASSWORD"),
        )

    def _execute_query(self, sql_file_path: Path) -> list[tuple[Any, ...]]:
        cur = self._connection.cursor()
        with sql_file_path.open(mode="r") as q:
            query = q.read()
        cur.execute(query=query)
        result = cur.fetchall()
        cur.close()
        return result

    def _write_query_to_csv(
        self, query_result: list[tuple[Any, ...]], query_gen_name: str
    ):
        output_path = self._output_dir / query_gen_name

        with output_path.open(mode="w", newline="") as q_out_file:
            writer = csv.writer(q_out_file)
            writer.writerow(query_result)

    def _run_query_and_save_to_csv(self, sql_file_path: Path):
        assert sql_file_path.name.endswith(".sql")
        query_gen_name = sql_file_path.name[: -len(".sql")]
        query_result = self._execute_query(sql_file_path=sql_file_path)
        self._write_query_to_csv(
            query_result=query_result, query_gen_name=query_gen_name
        )

    def run_sql_queries(
        self, sql_query_paths: list[Path]
    ) -> list[Path]:
        result_paths = []
        for query_path in sql_query_paths:
            self._run_query_and_save_to_csv(sql_file_path=query_path)
            assert query_path.exists()
            result_paths.append(query_path)
        return result_paths

    def close_connection(self):
        if self._connection is not None:
            self._connection.close()


