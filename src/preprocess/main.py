import pprint
import time
from prefilter import Prefilter
from icustay_measurement_combiner import ICUStayMeasurementCombiner
from sample_list_builder import FullAdmissionListBuilder
from feature_buillder import FeatureBuilder
from feature_finallizer import FeatureFinalizer


if __name__ == "__main__":
    start = time.time()

    print("Starting Prefilter")
    prefilter = Prefilter()
    prefilter_exports = prefilter()
    print("Prefilter Exports:")
    pprint.pprint(prefilter_exports, indent=4)
    print("Done with Prefilter\n")

    print("Starting ICUStatyMeasurementCombiner")
    measurement_combiner = ICUStayMeasurementCombiner()
    measurement_combiner_exports = measurement_combiner()
    print("ICUStatyMeasurementCombiner exports:")
    pprint.pprint(measurement_combiner_exports, indent=4)
    print("Done with ICUStatyMeasurementCombiner\n")

    print("Starting FullAdmissionListBuilder")
    hadm_list_builder = FullAdmissionListBuilder()
    hadm_list_builder_exports = hadm_list_builder()
    print("FullAdmissionListBuilder exports:")
    pprint.pprint(hadm_list_builder_exports, indent=4)
    print("Done with FullAdmissionListBuilder\n")

    print("Starting FeatureBuilder")
    feature_builder = FeatureBuilder()
    feature_builder_exports = feature_builder()
    print("FeatureBuilder exports:")
    pprint.pprint(feature_builder_exports, indent=4)
    print("Done with FeatureBuilder\n")

    print("Starting FeatureFinalizer")
    feature_finalizer = FeatureFinalizer()
    feature_finalizer_exports = feature_finalizer()
    print("FeatureFinalizer exports:")
    pprint.pprint(feature_builder_exports, indent=4)
    print("Done with FeatureFinalizer\n")

    end = time.time()

    print(f"All Done!\nTotal preprocessing time = {end - start}")
