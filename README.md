# ehr_adversarial_attack
A reproduction study of adversarial attacks on a LSTM time series model

## Overview
This project aims to reproduce results originally published in:

Sun, M., Tang, F., Yi, J., Wang, F., & Zhou, J. (2018, July). Identify susceptible locations in medical records via adversarial attacks on deep predictive models. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining* (pp. 793-801). (https://arxiv.org/abs/1802.04822)



## Dependencies

* Python 3.9+

* dill

* gensim

* matplotlib

* numpy

* pandas

* scikit-learn

* seaborn

* Docker

  

## Data Pipeline

This work uses a subset of the MIMIC-III database. The queries used to extract the necessary views and tables from MIMIC-III are saved in `ehr_adversarial_attack/src/docker_db/mimiciii_queries` . The outputs from running these queries are saved in `ehr_adversarial_attack/data/mimiciii_query_results`. The procedure below describes all steps required to preprocess these query outputs into data that can be accepted by an LSTM model, use these data to train the model, and then run adversarial attacks on the trained model. The outputs of each of these steps have already been saved in files noted below, so if you just want to see the final result of adversarial attacks, you can jump to the Attack Analysis Section.

### 1. Convert SQL Query Outputs to Model Input Data

Run file `src/preprocess/main.py`. This will generate input feature matrices for each patient and a vector of mortality labels (saved as `measurement_data_list.pickle ` and `in_hospital_mortality_list.pickle` in directory `ehr_adversarial_attack/output_feature_finalizer`.

### 2. Train the LSTM model

Run file `src/LSTMSun2018_cross_validate.py`. This will train LSTM model using cross-validation. Note that this method (used in the original paper) has drawbacks due to the fact that it never evaluates the model on unseen data, so there is significant risk of overfitting).

During training, model parameter checkpoint files will be saved in `data/cross_validate_sun2018_fulll48m19_01`.

### 3. Attack the trained model

Run file `src/adv_attack_full48_m19.py`. With current settings, this will run adversarial attacks on 1000 randomly selected samples, with oversampling of the minority `mortality = 1` class so that the number of attacked samples with `mortality = 0` and`mortality = 1` will be roughly equal. Results of of the adversarial attacks will be saved in `data/attack_results_f48_00`

### 4. Plot results of the adversarial attacks

Run file `src/attack_results_analyzer.py` to produce plots of the attack susceptibilities vs. time for each of the 19 measurement parameters.



## Key Results





![](https://github.com/duanegoodner/ehr_adversarial_attack/blob/main/data/images/Table.png)



![](https://github.com/duanegoodner/ehr_adversarial_attack/blob/main/data/images/plots.png)



