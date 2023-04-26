import numpy as np
import pandas as pd
import pickle
from pathlib import Path


prefiltered_output_dir = (
    Path(__file__).parent.parent.parent
    / "data"
    / "prefilter_output"
)

icustay_path = prefiltered_output_dir / "icustay_detail.pickle"
d_icd_diagnoses_path = prefiltered_output_dir / "d_icd_diagnoses.pickle"
diagnoses_icd_path = prefiltered_output_dir / "diagnoses_icd.pickle"


with icustay_path.open(mode="rb") as p:
    icustay = pickle.load(p)

with d_icd_diagnoses_path.open(mode="rb") as p:
    d_icd_diagnoses = pickle.load(p)

with diagnoses_icd_path.open(mode="rb") as p:
    diagnoses_icd = pickle.load(p)


labels_builder_output_dir = prefiltered_output_dir = (
    Path(__file__).parent.parent.parent
    / "data"
    / "labels_builder_output"
)

top_n_codes_path = labels_builder_output_dir / "top_25_diagnoses.pickle"

with top_n_codes_path.open(mode="rb") as p:
    top_n_codes = pickle.load(p)

