import pandas as pd

import preprocess_settings as ps





class InitialDataframeFilter:
    def __init__(self, filter_settings: ps.InitialFilterSettings):
        self._filter_settings = filter_settings

    def filter_icu_stay_detail(self) -> pd.DataFrame:

