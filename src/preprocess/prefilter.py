import pandas as pd

import preprocess_settings as ps


class Prefilter:
    def __init__(self, settings: ps.PrefilterSettings):
        self._settings = settings


    def filter(self, df: pd.DataFrame, ):


