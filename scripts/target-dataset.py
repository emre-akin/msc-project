###########################################################################################################
###  Mal-API-2019 dataset reference
#    Title: Deep learning based Sequential model for malware analysis using Windows exe API Calls
#    Author: Ahmet Faruk Yazı, Ferhat Ozgur Catak, Ensar Gul
#    Date: 2020
#    Availability: https://github.com/ocatak/malware_api_class
###########################################################################################################

import os
import pandas as pd

# Load datasets
train = pd.read_csv('target_train_numerized.txt', names=['calls', 'types'])
test = pd.read_csv('target_test_numerized.txt', names=['calls', 'types'])

# Combine test & train
df = train.append(test, ignore_index=True)

len(df)

# Change all characters to lower case
df['calls'] = df['calls'].str.lower()

# Limit maximum word size to 10000
temp_lst = list()
temp_str = str()

for x in range(len(df)):
    temp_lst = []
    temp_str = ''
    api_all = df.calls[x]
    api_list = api_all.split()
    for indx, i in enumerate(api_list):
        if indx == 10000:
            temp_str = ' '.join(temp_lst)
            df.calls[x] = temp_str
            break
        else:
            temp_lst.append(i)

# Create seperate columns for every label
df['Backdoor'] = 0
df['Trojan'] = 0
df['Spyware'] = 0
df['Worms'] = 0
df['Dropper'] = 0
df['Downloader'] = 0
df['Virus'] = 0
df['Adware'] = 0

for i in range(len(df)):
  if df.types[i] == 0:
    df.at[i, 'Backdoor'] = 1
  elif df.types[i] == 1:
    df.at[i, 'Trojan'] = 1
  elif df.types[i] == 2:
    df.at[i, 'Spyware'] = 1
  elif df.types[i] == 3:
    df.at[i, 'Worms'] = 1
  elif df.types[i] == 4:
    df.at[i, 'Dropper'] = 1
  elif df.types[i] == 5:
    df.at[i, 'Downloader'] = 1
  elif df.types[i] == 6:
    df.at[i, 'Virus'] = 1
  elif df.types[i] == 7:
    df.at[i, 'Adware'] = 1

df.head(3)

df.info()

# Change directory
os.chdir('../')

# Save
df.to_csv('target_final.txt', index=False, header=False)