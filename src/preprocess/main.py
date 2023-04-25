import preprocess_settings as ps
import prefilter_old as pf
import preprocessor as pr


def main():
    prefilter = pf.Prefilter(
        settings=ps.DEFAULT_SETTINGS.prefilter_settings
    )
    preprocessor = pr.Preprocessor(
        prefilter=prefilter,
        icustay_detail_csv=ps.DEFAULT_SETTINGS.icustay_detail_csv,
        diagnoses_icd_csv=ps.DEFAULT_SETTINGS.diagnoses_icd_csv,
        d_icd_diagnoses_csv=ps.DEFAULT_SETTINGS.d_icd_diagnoses_csv
    )
    result = preprocessor.run_prefilter()


if __name__ == "__main__":
    main()
