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





 



All of the procedures described below have already been run, and their outputs have been saved in files as noted. If you just want to plot the results of adversarial attacks on a trained model you can skip to the Attack Analysis section.

### A. Building a Local MIMIC-III Database

1. Create directory `mimiciii_raw/csv/` under the `ehr_adversarial_attack` root directory.

2. If you do not have a physionet account and/or "credentialed" physionet access, you go [here](https://physionet.org/register/) and [here](https://physionet.org/settings/credentialing/) to obtain them.

2.  Follow the instructions at https://physionet.org/content/mimiciii/1.4/ to download the database .gz. 

3. Extrac all of the .gz files into .csv files. Place the extracted .csv files in `ehr_adversarial_attack/mimiciii_raw/csv/`

4. From directory `ehr_adversarial_attack/src/docker_db/` run the commands:

   ````
   $ docker compose build
   $ docker compose up
   ````

   Running `docker compose up` will start a Docker container that will build a MIMIC-III database (inside the container). This will likely take ~30 minutes.

5.  From directory `ehr_adversarial_attack/src/docker_db/` run:

   ```
   $ ./run_queries.sh
   ```

   



This section describes the full procedure necessary to obtain the dataset, pre-process data, train the LSTM model, perform adversarial attacks, and view the results of these attacks. Each of these steps have already run, and their outputs have been saved in files noted below.

### Building a Local MIMIC-III Database



### Pre-processing





## Training the LSTM Model





## Dataset

This work uses a subset of the MIMIC-III Clinical Database. The database queries and additional preprocessing steps necessary to prepare data for input into the LSTM model have already been performed, 









Data Preprocessing Details

The following preprocessing steps have already been perfomed to produce files needed for model training input. (So you don't need to run them, but leaving the steps here for documentation)

and converted to the feature and label tensors needed for the LSTM model and saved as files `measurement_data_list.pickle ` and `in_hospital_mortality_list.pickle` in directory ``ehr_adversarial_attack/data/output_feature_finalizer/`.



The queries providing the necessary tables and views from MIMIC-III have already been run (by creating a PostgreSQL database inside a Docker container defined ``` ), and their output is saved in `ehr_adversarial_attack/mimiciii_query_results/`. The data have been further processed into feature and response 

* 

* Building a local MIMIC-III datablase:

  1. 

  2. 

  

Preprocessing



Training



Results









