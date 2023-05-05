import numpy as np
import pandas as pd

# create sample data
df = pd.DataFrame(np.random.randn(5, 4))
df.iloc[1, 1] = np.nan
df.iloc[2, 3] = np.nan

# columns to winsorize
cols_to_winsorize = [0, 1, 2, 3]

# calculate the 5th and 95th percentiles for each column
q05 = df[cols_to_winsorize].quantile(0.05)
q95 = df[cols_to_winsorize].quantile(0.95)

# replace values outside the limits with the 5th or 95th percentile values
df[cols_to_winsorize] = df[cols_to_winsorize].clip(q05, q95, axis=1)

# replace NaN values with the median of each column to winsorize
# df[cols_to_winsorize] = df[cols_to_winsorize].fillna(
#     df[cols_to_winsorize].median()
# )

# print the modified dataframe
print(df)
