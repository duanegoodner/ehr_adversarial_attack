import hashlib
import pyarrow as pa
import pandas as pd
from typing import NamedTuple
import data_io as dfp

df = pd.DataFrame({"a": [1, 2, 3]})
# Convert from pandas to Arrow
table = pa.Table.from_pandas(df)

diag_icd_pd = dfp.DEFAULT_DATAFRAME_PROVIDER.import_query_result(
    query_name="diagnoses_icd"
)


class NamedDataframe(NamedTuple):
    names: list[str]
    df: pd.DataFrame


my_named_df = NamedDataframe(names=["Joe"], df=df)




