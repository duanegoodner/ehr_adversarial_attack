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

plt.subplot(411)
plt.imshow(dataset[0][0], cmap='RdYlBu_r')
plt.subplot(412)
plt.imshow(dataset[1][0], cmap='RdYlBu_r')
plt.subplot(413)
plt.imshow(dataset[2][0], cmap='RdYlBu_r')
plt.subplot(414)
plt.imshow(dataset[3][0], cmap='RdYlBu_r')

cax = plt.axes([0.85, 0.1, 0.075, 0.8])
# plt.colorbar(cax=cax)

plt.show()
