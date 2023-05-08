from pathlib import Path
from adv_attack import AdversarialExamplesSummary
import preprocess.resource_io as rio


result_dir = Path(__file__).parent.parent / "data" / "attack_results_f48_01"

importer = rio.ResourceImporter()
result = importer.import_pickle_to_object(path=result_dir / "k0.0-l10.1-lr0"
                                                            ".1-ma100-ms1"
                                                            "-2023-05-08_10"
                                                            ":28:49.932802"
                                                            ".pickle")

