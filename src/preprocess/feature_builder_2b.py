import dill
import resource_io as rio
from pathlib import Path



pickle_path = Path(
    "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
    "/output_full_admission_list_builder/full_admission_list.pickle"
)

full_admission_data = dill.load(pickle_path.open(mode="rb"))


