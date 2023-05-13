import resource_io as rio

import prefilter_input_classes as pfin


features_path = pfin.FEATURE_FINALIZER_OUTPUT_DIR / "measurement_data.pickle"
in_hosp_mortality_path = (
    pfin.FEATURE_FINALIZER_OUTPUT_DIR / "in_hospital_mortality.pickle"
)
col_name_path = (
    pfin.FEATURE_FINALIZER_OUTPUT_DIR / "measurement_col_names.pickle"
)

importer = rio.ResourceImporter()
features = importer.import_pickle_to_object(path=features_path)
in_hosp_mortality = importer.import_pickle_to_object(
    path=in_hosp_mortality_path
)
col_names = importer.import_pickle_to_object(path=col_name_path)

