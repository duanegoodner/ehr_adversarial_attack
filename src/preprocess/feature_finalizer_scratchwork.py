import dill
import numpy as np
import pandas as pd
import time

import resource_io as rio
from pathlib import Path


partly_processed_hadm_list_path = Path(
    "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
    "/output_feature_builder/hadm_list_with_processed_dfs.pickle"
)

with partly_processed_hadm_list_path.open(mode="rb") as f:
    hadm_list = dill.load(f)

x19 = []
y = []


start = time.time()

for idx in range(10):
    cur_item = hadm_list[idx]

    data_in_cutoff_time = cur_item.time_series[
        (cur_item.time_series["charttime"] >= cur_item.admittime[0]) &
        (
            cur_item.time_series["charttime"]
            < cur_item.admittime[0] + pd.Timedelta(hours=48)
        )
    ]
    data_in_cutoff_time = data_in_cutoff_time.drop(["charttime"], axis=1)

    x19.append(data_in_cutoff_time.values)

    y.append(hadm_list[idx].hospital_expire_flag)

end = time.time()

print(f"Time to process 10 samples: {end - start}")
