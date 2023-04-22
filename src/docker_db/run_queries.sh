#!/bin/bash

source .env
docker compose up -d
docker exec -it mimic_db "$CONTAINER_SQL_DIR"/run_queries_container.sh
docker compose down
sudo chown -R "$USER":"$USER" "$LOCAL_OUTPUT_CSV" "$LOCAL_DB"

orig_dir=$PWD

cd "$LOCAL_OUTPUT_CSV" || exit

for file in *.csv; do
  echo Buildig tar.gz archive of "$file"
  tar -zcvf "$file".tar.gz "$file";
done

cd "$orig_dir" || exit


