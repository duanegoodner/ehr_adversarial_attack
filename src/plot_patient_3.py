import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


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

fig, axes = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(6, 6))
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
plt.subplots_adjust(right=0.8)

for i, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax2 = ax.twinx()
    ax2.tick_params(axis='y', length=0, labelsize=0, labelcolor='white')
    ax2.set_ylabel("M = ?", rotation=0, labelpad=15)


plt.show()


