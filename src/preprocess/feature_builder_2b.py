import dill
import resource_io as rio
from pathlib import Path
from feature_builder_2 import FullAdmissionData


pickle_path = Path(
    "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
    "/output_full_admission_objects/full_admisssion.pickle"
)

full_admission_data = dill.load(pickle_path.open(mode="rb"))


