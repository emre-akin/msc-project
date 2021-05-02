#!/bin/bash

# Download data sets
cd scripts/
./download.sh

# API-Call dataset data processing
cd ../datasets/api-call-dataset/
python ../../scripts/api-call-dataset.py mal-api-2019.zip
python ../../target-dataset.py

# Ember Dataset data processing
cd ../ember-dataset/ember2018/
python ../../../scripts/ember-dataset.py

# Delete empty lines
sed -i '/^[[:space:]]*$/d' pre-train.txt

python ../../../scripts/source-dataset.py

# Create the word embeddings
cd ../../../
mkdir embeddings

python scripts/source-glove.py
python scripts/source-glove-finetune.py
python scripts/target-glove.py

# Train the models
mkdir models
cd models/
mkdir 256_bilstm
mkdir tf
mkdir tf_finetune
mkdir 200epoch
mkdir 256_lstm
cd ../

python scripts/train.py