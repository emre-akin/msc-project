#!/bin/bash
###########################################################################################################
###  EMBER dataset reference
#    Title: EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models
#    Author: Hyrum S. Anderson, Phil Roth
#    Date: 2018
#    Availability: https://github.com/elastic/ember
###########################################################################################################
###  Mal-API-2019 dataset reference
#    Title: Deep learning based Sequential model for malware analysis using Windows exe API Calls
#    Author: Ahmet Faruk Yazı, Ferhat Ozgur Catak, Ensar Gul
#    Date: 2020
#    Availability: https://github.com/ocatak/malware_api_class
###########################################################################################################

cd ..
mkdir datasets
cd datasets

mkdir ember-dataset
cd ember-dataset

curl -o ember_dataset_2018_2.tar.bz2 https://ember.elastic.co/ember_dataset_2018_2.tar.bz2

tar -xf ember_dataset_2018_2.tar.bz2

###################

cd ..
mkdir api-call-dataset
cd api-call-dataset

curl -o mal-api-2019.zip https://raw.githubusercontent.com/ocatak/malware_api_class/master/mal-api-2019.zip

curl -o labels.csv https://raw.githubusercontent.com/ocatak/malware_api_class/master/labels.csv