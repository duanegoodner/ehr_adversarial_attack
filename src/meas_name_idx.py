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
