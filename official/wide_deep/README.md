# Predicting Income with the Census Income Dataset

There are two samples provided in this directory. Both allow you to move from
single-worker training to distributed training without any code changes, and
make it easy to export model binaries for prediction, but with the following
distinction:


* The sample provided in [Low Level TensorFlow](./tflowlevel) uses the low level
  bindings to build a model. This example is great for understanding the
  underlying workings of TensorFlow and best practices when using the low-level
  APIs.

* The sample provided in [Estimator](./estimator) uses the high level
  `tf.contrib.learn.Estimator` API. This API is great for fast iteration and
  quickly adapting models to your own datasets without major code overhauls.

All the models provided in this directory can be run on the Cloud Machine Learning Engine. To follow along, check out the setup instructions [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

## Download the data
The [Census Income Data
Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) that this sample
uses for training is hosted by the [UC Irvine Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets/). We have hosted the data
on Google Cloud Storage in a slightly cleaned form:

 * Training file is `adult.data.csv`
 * Evaluation file is `adult.test.csv`

### Disclaimer
The source of this dataset is from a third party. Google provides no representation,
warranty, or other guarantees about the validity or any other aspects of this dataset.

### Set Environment Variables
Please run the export and copy statements first:

```
GCS_TRAIN_FILE=gs://cloudml-public/census/data/adult.data.csv
GCS_EVAL_FILE=gs://cloudml-public/census/data/adult.test.csv

CENSUS_DATA=census_data
mkdir $CENSUS_DATA

TRAIN_FILE=`pwd`/$CENSUS_DATA/adult.data.csv
EVAL_FILE=`pwd`/$CENSUS_DATA/adult.test.csv

gsutil cp $GCS_TRAIN_FILE $TRAIN_FILE
gsutil cp $GCS_EVAL_FILE $EVAL_FILE
```

## Virtual environment
Virtual environments are strongly suggested, but not required. Installing this
sample's dependencies in a new virtual environment allows you to run the sample
without changing global python packages on your system.

There are two options for the virtual environments:
 * Install [Virtual](https://virtualenv.pypa.io/en/stable/) env
   * Create virtual environment `virtualenv single-tf`
   * Activate env `source single-tf/bin/activate`
 * Install [Miniconda](https://conda.io/miniconda.html)
   * Create conda environment `conda create --name single-tf python=2.7`
   * Activate env `source activate single-tf`


## Install dependencies
Install the dependencies using `pip install -r requirements.txt`

## High Level and Core TensorFlow
You have an option of running either the High level estimator code or Core
TensorFlow. Depending on which version you want to run:

* High Level Estimator requires `cd estimator`

* Core TensorFlow requires `cd tensorflowcore`

# Single Node Training
Single node training runs TensorFlow code on a single instance. You can run the exact
same code locally and on Cloud ML Engine.

## How to run the code
You can run the code either as a stand-alone python program or using `gcloud`.
See options below:

### Using local python
Run the code on your local machine:

```
export TRAIN_STEPS=5000
export OUTPUT_DIR=census_output
rm -rf $OUTPUT_DIR
```

```
python trainer/task.py --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --job-dir $OUTPUT_DIR \
                       --train-steps $TRAIN_STEPS \
                       --eval-steps 100
```

### Using local trainer
In order to run the code locally using gcloud see
[here](https://cloud.google.com/ml-engine/docs/how-tos/getting-started-training-prediction#local-train-single)

## Tensorboard
Run the Tensorboard to inspect the details about the graph.

```
tensorboard --logdir=$GCS_JOB_DIR
```

## Accuracy and Output
You should see the output for default number of training steps and approx accuracy close to `80%`.

# Distributed Node Training
Distributed node training uses [Distributed
TensorFlow](https://www.tensorflow.org/deploy/distributed). Please see how to
run it locally
[here](https://cloud.google.com/ml-engine/docs/how-tos/getting-started-training-prediction#local-train-dist)

## How to run the code
In order to run the code on Cloud ML Engine see
[here](https://cloud.google.com/ml-engine/docs/how-tos/getting-started-training-prediction#cloud-train-single)

# Hyperparameter Tuning
[Google Cloud ML Engine](https://cloud.google.com/ml-engine/) allows you to perform Hyperparameter tuning to find out the
most optimal hyperparameters. See [Overview of Hyperparameter Tuning](https://cloud.google.com/ml/docs/concepts/hyperparameter-tuning-overview) for more details.

## Running Hyperparameter Job
In order to run Hyperparameter tuning on Cloud ML Engine see
[here](https://cloud.google.com/ml-engine/docs/how-tos/getting-started-training-prediction#hyperparameter_tuning)

## Run Predictions on CMLE
In order to run prediction on Cloud ML Engine see
[here](https://cloud.google.com/ml-engine/docs/how-tos/getting-started-training-prediction#deploy_a_model_to_support_prediction)
