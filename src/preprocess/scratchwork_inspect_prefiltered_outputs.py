import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import resource_io as rio

prefiltered_output_dir = (
    Path(__file__).parent.parent.parent / "data" / "prefilter_output"
)

importer = rio.ResourceImporter()


admissions = importer.import_resource(
    prefiltered_output_dir / "admissions.pickle"
)


icustay = importer.import_resource(
    prefiltered_output_dir / "icustay_detail.pickle"
)

stays = pd.merge(
    left=admissions, right=icustay, on="hadm_id", suffixes=("_h", "_i")
)




# d_icd_diagnoses = importer.import_resource(
#     prefiltered_output_dir / "d_icd_diagnoses.pickle"
# )
#
# diagnoses_icd = importer.import_resource(
#     prefiltered_output_dir / "diagnoses_icd.pickle"
# )
#
# pivoted_bg = importer.import_resource(
#     prefiltered_output_dir / "pivoted_bg.pickle"
# )
#
# pivoted_lab = importer.import_resource(
#     prefiltered_output_dir / "pivoted_lab.pickle"
# )
#
# pivoted_vital = importer.import_resource(
#     prefiltered_output_dir / "pivoted_vital.pickle"
# )
