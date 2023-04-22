import pandas as pd
import preprocess_settings as ps


class DataFrameProvider:
    def __init__(self, settings: ps.PreprocessSettings = ps.DEFAULT_SETTINGS):
        self.settings = settings
        assert settings.query_result_format == ".csv"

    # TODO add methods to handle other raw formats; then relax csv requirement
    def import_query_result(self, query_base_name: str):
        assert query_base_name in self.settings.query_base_names
        df = pd.read_csv(
            f"{self.settings.mimiciii_query_results}/{query_base_name}.csv"
        )
        df.columns = [item.lower() for item in list(df.columns)]
        return df


DEFAULT_DATAFRAME_PROVIDER = DataFrameProvider()


if __name__ == "__main__":
    df_provider = DataFrameProvider()
    admissions_df = df_provider.import_query_result("admissions")
    admissions_np = admissions_df.to_numpy()





