import pandas as pd
import numpy as np

# create a sample dataframe with irregular timestamps
df = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(
            [
                "2023-05-04 00:00:01",
                "2023-05-04 00:15:32",
                "2023-05-04 00:48:12",
                "2023-05-04 01:03:05",
                "2023-05-04 01:29:22",
                "2023-05-04 02:15:10",
                "2023-05-04 03:02:08",
                "2023-05-04 04:16:39",
                "2023-05-04 04:45:17",
            ]
        ),
        "measurement1": [1, 2, 3, np.nan, 5, np.nan, 7, 8, np.nan],
        "measurement2": [11, np.nan, 13, 14, 15, 16, np.nan, 18, 19],
        "measurement3": [21, 22, 23, 24, 25, 26, 27, 28, 29],
    }
)

# set the timestamp column as the index
df.set_index("timestamp", inplace=True)

# resample the dataframe to hourly intervals using the mean function
df_resampled = df.resample("H").mean()

print(df_resampled)
