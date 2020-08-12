GCP Setup
=========

# Provisioning 

    export PROJECT_ID=...
    export BUCKET=...

    gcloud config set project $PROJECT_ID
    gcloud config set compute/zone us-central1-b
    gcloud config set compute/region us-central1

    gsutil mb -c standard -l us-central1 -b on gs://$BUCKET
    ctpu up --name ml-1 --preemptible 

    gcloud compute ssh ml-1

# Training

    export MODEL=lstm_v3 && \
    export BUCKET=... && \
    export TPU_NAME=ml-1 && \
    export STORAGE_BUCKET=gs://$BUCKET && \
    export MODEL_DIR=$STORAGE_BUCKET/$MODEL && \
    export DATA_DIR=$STORAGE_BUCKET/data && \
    export PYTHONPATH="$PYTHONPATH:/usr/share/models"

    python3 $MODEL.py \
    --model_dir=$MODEL_DIR \
    --data_dir=$DATA_DIR \
    --distribution_strategy=tpu \
    --tpu=$TPU_NAME \
    --train_epochs=100

# Resources

- console
    - https://console.cloud.google.com
- tutorials
    - https://cloud.google.com/tpu/docs/using-estimator-api
    - https://cloud.google.com/tpu/docs/tutorials
    - https://blog.tensorflow.org/2018/08/code-with-eager-execution-run-with-graphs.html
    - https://www.tensorflow.org/guide/migrate
    - https://github.com/tensorflow/tpu/tree/master/models/official
    - https://github.com/tensorflow/models/tree/master/official
    - https://blog.tensorflow.org/2019/01/what-are-symbolic-and-imperative-apis.html
    - https://developers.google.com/machine-learning/guides/text-classification/step-3
    - https://pbpython.com/categorical-encoding.html
    