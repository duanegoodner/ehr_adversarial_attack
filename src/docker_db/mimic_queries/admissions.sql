copy (
    select *
    from mimiciii.admissions
) to '/mimic_query_results/admissions.csv' with delimiter ',' csv header;