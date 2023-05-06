import pandas as pd
import numpy as np

# create a sample dataframe with timestamp values and measurement data
np.random.seed(0)
df = pd.DataFrame({
    'timestamp': pd.date_range('2022-01-01', periods=50, freq='H'),
    'measurement': np.random.rand(50) * 100
})

# set the timestamp column as the index
df.set_index('timestamp', inplace=True)

# resample the dataframe to daily OHLC values
df_resampled = df.resample('D').ohlc()

print(df_resampled)
