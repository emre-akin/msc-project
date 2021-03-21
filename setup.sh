#!/bin/bash

# Download data sets
cd scripts/
./download.sh

# API-Call data set data processing
cd ../datasets/api-call-dataset/
python ../../scripts/api-call-dataset.py mal-api-2019.zip

# Ember Data set data processing & feature extraction

cd ../ember-dataset/ember2018/
python ../../../scripts/ember-dataset.py

sed -i '/^[[:space:]]*$/d' pre-train.txt