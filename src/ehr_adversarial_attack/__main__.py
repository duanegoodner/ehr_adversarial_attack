from pathlib import Path
from mimiciii_database import MimiciiiDatabaseAccess


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    db_dotenv_path = project_root / "config" / "mimiciii_database.env"
    db_output_dir = project_root / "data" / "mimiciii_query_results_psycopg2"

    db_access = MimiciiiDatabaseAccess(
        dotenv_path=db_dotenv_path, output_dir=db_output_dir
    )

    query_dir = project_root / "src" / "mimiciii_queries"
    query_filenames = [
        "icustay_detail.sql",
        "pivoted_bg.sql",
        "pivoted_lab.sql",
        "pivoted_vital.sql"
    ]
    query_paths = [query_dir / filename for filename in query_filenames]

    db_access.connect()
    db_access.run_sql_queries(sql_query_paths=query_paths)
