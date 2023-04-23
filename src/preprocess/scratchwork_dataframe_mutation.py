import pandas as pd


class NamedDataframe:
    def __init__(self, df: pd.DataFrame, name: str):
        self.df = df
        self.name = name

    def import_by_simple_return(self):
        return self.df

    def import_by_deep_copy(self):
        return self.df.copy(deep=True)

    def import_by_shallow_copy(self):
        return self.df.copy(deep=False)


my_df = pd.DataFrame({"a": [1, 2, 3]})

named_df = NamedDataframe(df=my_df, name="Joe")

my_imported_df_by_return = named_df.import_by_simple_return()
my_imported_df_by_deep_copy = named_df.import_by_copy()


