# ehr_adversarial_attack
*A reproduction study of adversarial attacks on a LSTM time series model*



## Overview

This project aims to reproduce results originally published in:

Sun, M., Tang, F., Yi, J., Wang, F. and Zhou, J., 2018, July. Identify susceptible locations in medical records via adversarial attacks on deep predictive models. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining* (pp. 793-801).  (https://arxiv.org/abs/1802.04822)

The original paper trained a Long Short-Term Memory (LSTM) model using time series of Intensive Care Unit (ICU) patient vital-sign and lab measurements as the inputs, and in-hospital mortality as the prediction target. An adversarial attack algorithm was then used to identify small perturbations which, when applied to a real, correctly-classified input sample, result in misclassification of the perturbed input. Susceptibility calculations were then performed to quantify the attack vulnerability as functions of time and the measurement type within the input feature space.



## Original Paper Code Repository

The original authors did not publish a code repository for this particular work, but some of the authors reported on predictive modeling (but not adversarial attack) with a similar LSTM in:

Tang, F., Xiao, C., Wang, F. and Zhou, J., 2018. Predictive modeling in urgent care: a comparative study of machine learning approaches. *Jamia Open*, *1*(1), pp.87-98.

The repository for this paper is available at: https://github.com/illidanlab/urgent-care-comparative



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

  

## Instructions for running code and viewing results in this repository

#### Step 1. Clone a local copy of this project repo:

```
$ https://github.com/duanegoodner/ehr_adversarial_attack
```



### **Step 2. Download data**

Obtain credentialed access to Physionet.org, and run the following command from your terminal:

```
$ wget -r -N -c -np --user <your_physionet_username> --ask-password https://physionet.org/files/mimiciii/1.4/
```



### Step 3. Extract MIMIC-III csv files

Extract the file downloaded in step 2. Save all expanded .csv files to `ehr_adversarial_attack/data/mimiciii_raw/csv/`



### Step 3.  Build a Postgres MIMIC-III database using Docker

From directory `ehr_adversarial_attack/src/docker_db`, run the following commands:

```
$ docker compose build
$ docker compose up
# Details of the database build process taking place inside a container will show in the terminal. Will take ~45 minutes
```

Once the database has finished building, use CTRL-C to stop the container.



### Step 4. Run the MIMIC-III queries

From directory `ehr_adversarial_attack/src/docker_db`, run:

```
$ ./run_queries.sh
```

 This will re-start the Docker container, run the necessary .sql queries, and save the results to `ehr_adversarial_attack/data/mimiciii_query_results`.



### Step 5. Pre-process the data

From the repository root directory, run the following command to preprocess the SQL query outputs into the LSTM model input format:

```
$ python src/preprocess/main.py
```

Output:

```shell
Starting Prefilter
Done with Prefilter

Starting ICUStatyMeasurementCombiner
Done with ICUStatyMeasurementCombiner

Starting FullAdmissionListBuilder
Done with FullAdmissionListBuilder

Starting FeatureBuilder
Done processing sample 0/41960
Done processing sample 5000/41960
Done processing sample 10000/41960
Done processing sample 15000/41960
Done processing sample 20000/41960
Done processing sample 25000/41960
Done processing sample 30000/41960
Done processing sample 35000/41960
Done processing sample 40000/41960
Done with FeatureBuilder

Starting FeatureFinalizer
Done with FeatureFinalizer

All Done!
Total preprocessing time = 555.2803893089294 seconds

```



### Step 3. Obtain a trained LSTM model

From the repository root directory, run:

```
$ python src/LSTMSun2018_cross_validate.py
```

This will train LSTM model using cross-validation and and evaluate the predictive performance periodically throughout the training process. Note that this method (used in the original paper) has drawbacks due to the fact that it never evaluates the model on unseen data, so there is significant risk of overfitting.

Output:

```ll
# Many realtime updates on training loss and evaluation metrics
Model parameters saved in file: 2023-05-07_23:40:59.241609.tar
# your filename will be different, but will have the same format
```

Take note of the .tar filename. for use in Step 4.



### Step 4. Run adversarial attacks on the model

Now we are ready to run an adversarial attack on an the trained LSTM model. From the repository root, run:

```
$ python src/adv_attack_full48_m19.py -f <file_name_obained_from_step_3.tar>
```

Example Output:

```
# Updates on elapsed time and index of sample under attack
Attack summary data saved in k0.0-l10.15-lr0.1-ma100-ms1-2023-05-07_23:47:51.261865.pickle
# Your .tar filename will be different, but will have the same format.
```

Once again, only the filename (not full path), is printed, but that's all you need for the next step.



### Step 5. Plot results of the adversarial attacks

From the repo root directory, run:

```
$ python src/attack_results_analyzer.py -f <filename_from_step4_output.tar>
```

For example, if your output from Step 4 said: `Attack summary data saved in k0.0-l10.15-lr0.1-ma100-ms1-2023-05-07_23:47:51.261865.pickle`, you would run `$ python src/attack_results_analyzer.py -f k0.0-l10.15-lr0.1-ma100-ms1-2023-05-07_23:47:51.261865.pickle `

This will produce plots of attack susceptibilities of various measurements vs. time for 0-to-1 and 1-to-0 attacks.  (The plot for 1-to-0 should appear first. Depending on your environment, you may need to close this plot window before the 0-to-1 plot appears.)



## Key Results

#### LSTM predictive metrics

Here are the LSTM predictive evaluation metrics from the original paper and our work. The performance metric scores from our study are actually higher than those in the original work.

<img src="https://github.com/duanegoodner/ehr_adversarial_attack/blob/main/data/images/LStM_predictive_metrics.png"  width="40%" height="30%">



#### Adversarial attack susceptibility vs. measurement parameter

The table below indicates that in our study, we were unable to reproduce the attack susceptibilities reported in the original paper.

![](https://github.com/duanegoodner/ehr_adversarial_attack/blob/main/data/images/Table.png)



#### Adversarial attack susceptibility vs measurement time

These below plots also do NOT show the increase in susceptibility at later measurement times that were reported in the original paper.

![](https://github.com/duanegoodner/ehr_adversarial_attack/blob/main/data/images/plots.png)



