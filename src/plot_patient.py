import matplotlib.pyplot as plt
import pickle
import torch
from pathlib import Path
import numpy as np
import seaborn as sns
import project_config as pc


from x19_mort_dataset import X19MortalityDataset


varname_to_index = {
    "heartrate": 0,
    "sysbp": 1,
    "diasbp": 2,
    "tempc": 3,
    "resprate": 4,
    "spo2": 5,
    "glucose": 6,
    "albumin": 7,
    "bun": 8,
    "creatinine": 9,
    "sodium": 10,
    "bicarbonate": 11,
    "platelet": 12,
    "inr": 13,
    "potassium": 14,
    "calcium": 15,
    "ph": 16,
    "pco2": 17,
    "lactate": 18,
}
index_to_varname = {val: key for key, val in varname_to_index.items()}

dataset = X19MortalityDataset()
first_patient = dataset[0]

fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)
cbar_ax = fig.add_axes([.92, .3, .02, .4])

sns.heatmap(
    dataset[0][0],
    ax=axes[0],
    cbar=False,
    cmap="RdYlBu_r",
)

sns.heatmap(
    dataset[1][0],
    ax=axes[1],
    cbar=False,
    cmap="RdYlBu_r",
)

sns.heatmap(
    dataset[2][0],
    ax=axes[2],
    cbar=False,
    cmap="RdYlBu_r",
)

sns.heatmap(
    dataset[3][0],
    ax=axes[3],
    cbar=True,
    cmap="RdYlBu_r",
    cbar_ax=cbar_ax
)

fig.text(0.5, 0.02, "Time (hours)", ha="center")
fig.text(0.04, 0.5, "Feature Index", va="center", rotation="vertical")
for ax in axes:
    ax.set_xticks(range(0, 48, 5))
    ax.set_xticklabels([i for i in range(48) if i % 5 == 0])

plt.show()


