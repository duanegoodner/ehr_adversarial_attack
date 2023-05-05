import pandas as pd
import scipy.stats.mstats as mstats

# create a sample dataframe with NaN values
df = pd.DataFrame(
    {
        "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float("nan")],
        "col2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "col3": [22, 23, 24, 25, 26, 27, 28, 29, 30, float("nan"), 32],
        "col4": [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    }
)

# compute the 5th and 95th percentiles for each column
q05 = df.quantile(0.05)
q95 = df.quantile(0.95)

# winsorize each column
df_wins = df.apply(
    lambda x: (
        mstats.winsorize(x, limits=[0.05, 0.05]) if x.dtype != "object" else x
    )
)

# replace NaN values with the median of each column
# df_wins = df_wins.fillna(df_wins.median())

# print the original and winsorized dataframes
print("Original DataFrame:\n", df)
print("\nWinsorized DataFrame:\n", df_wins)
