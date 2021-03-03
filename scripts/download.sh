#!/bin/bash
# Please reference the datasets before using
# ember dataset: https://github.com/elastic/ember
# api-call dataset: https://github.com/ocatak/malware_api_class

cd ..
mkdir datasets
cd datasets

mkdir ember-dataset
cd ember-dataset

curl -o ember_dataset_2018_2.tar.bz2 https://ember.elastic.co/ember_dataset_2018_2.tar.bz2

###################

cd ..
mkdir api-call-dataset
cd api-call-dataset

curl -o mal-api-2019.zip https://raw.githubusercontent.com/ocatak/malware_api_class/master/mal-api-2019.zip