#!/bin/bash

sudo chown -R postgres:postgres /mimic_query_results

for file in /mimic_queries/*.sql; do
  echo Running query "$file"
  psql -U postgres -d mimic -f "$file"
done
