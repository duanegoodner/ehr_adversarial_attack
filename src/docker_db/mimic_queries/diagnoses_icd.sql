copy (
select * from mimiciii.diagnoses_icd
) to '/mimic_query_results/diagnoses_icd.csv' with delimiter ',' csv header;