import numpy as np
import pandas as pd
import pickle
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

bg_identified = pd.merge(
    left=icustay,
    right=bg,
    on=["hadm_id"],
    how="right",
    suffixes=("_icu", "_bg"),
)[
    [
        "subject_id",
        "hadm_id",
        "icustay_id_icu",
        "icustay_id_bg",
        "charttime",
        "potassium",
        "calcium",
        "ph",
        "pco2",
        "lactate",
    ]
]

lab_identified = pd.merge(
    left=icustay,
    right=lab,
    on=["hadm_id"],
    how="right",
    suffixes=("_icu", "_lab"),
).rename(columns={"subject_id_icu": "subject_id"})[
    [
        "subject_id",
        "hadm_id",
        "icustay_id_icu",
        "icustay_id_lab",
        "charttime",
        "albumin",
        "bun",
        "creatinine",
        "sodium",
        "bicarbonate",
        "glucose",
        "inr",
    ]
]

vital_identified = pd.merge(left=icustay, right=vital, on=["icustay_id"])[
    [
        "subject_id",
        "hadm_id",
        "icustay_id",
        "charttime",
        "heartrate",
        "sysbp",
        "diasbp",
        "tempc",
        "resprate",
        "spo2",
        "glucose",
    ]
]





