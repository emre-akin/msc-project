###########################################################################################################
###  Mal-API-2019 dataset reference
#    Title: Deep learning based Sequential model for malware analysis using Windows exe API Calls
#    Author: Ahmet Faruk Yazı, Ferhat Ozgur Catak, Ensar Gul
#    Date: 2020
#    Availability: https://github.com/ocatak/malware_api_class
###########################################################################################################

import sys
from zipfile import PyZipFile
import pandas as pd

# Unzip referance: https://docs.python.org/3/library/zipfile.html
for zipLocation in sys.argv[1:]:
    pzf = PyZipFile(zipLocation)
    pzf.extractall()

# Import dataset
df = pd.read_csv("all_analysis_data.txt", names=["API Calls"])
# Import labels
labels = pd.read_csv("labels.csv", names=["Malware Type"])

assert len(df) == len(labels)

# Combine dataframes
df = df.join(labels)

# Factorise labels
df['Malware Type'] = pd.factorize(df['Malware Type'])[0]

df.head()
df.describe()

# Create train and test data sets / 90% - 10%
train=df.sample(frac=0.9,random_state=200) #random state is a seed value
test=df.drop(train.index)

assert len(df) == len(train) + len(test)

# Save train and test data sets
train.to_csv('target_train_numerized.txt', index=False, header=False)
test.to_csv('target_test_numerized.txt', index=False, header=False)