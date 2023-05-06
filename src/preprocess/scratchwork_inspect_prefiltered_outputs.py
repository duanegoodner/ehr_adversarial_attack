import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
import resource_io as rio
import scipy.stats.mstats as mstats

prefiltered_output_dir = (
    Path(__file__).parent.parent.parent / "data" / "prefilter_output"
)

importer = rio.ResourceImporter()
icustay = importer.import_resource(prefiltered_output_dir / "icustay.pickle")
bg = importer.import_resource(prefiltered_output_dir / "bg.pickle")
lab = importer.import_resource(prefiltered_output_dir / "lab.pickle")
vital = importer.import_resource(prefiltered_output_dir / "vital.pickle")


pandas_start = time.time()
pandas_bg_groupby = bg.groupby(["hadm_id"])
pandas_end = time.time()

print(f"Pandas time = {pandas_end - pandas_start}")

numpy_start = time.time()
groups, indices = np.unique(bg["hadm_id"], return_inverse=True)
numpy_end = time.time()

print(f"Numpy time = {numpy_end - numpy_start}")


