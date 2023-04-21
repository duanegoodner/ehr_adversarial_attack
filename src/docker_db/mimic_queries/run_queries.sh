#!/bin/bash

sudo chown postgres:postgres /mimic_query_results

for file in /mimic_queries/*.sql; do
  psql -U postgres -d mimic -f "$file"
done