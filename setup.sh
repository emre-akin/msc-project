#!/bin/bash

# Download data sets
cd scripts/
./datasets.sh

# API-Call-data set data processing
cd ../datasets/api-call-dataset/
python ../../scripts/api-call-dataset.py mal-api-2019.zip