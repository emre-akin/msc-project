###########################################################################################################
###  EMBER dataset reference
#    Title: EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models
#    Author: Hyrum S. Anderson, Phil Roth
#    Date: 2018
#    Availability: https://github.com/elastic/ember
###########################################################################################################

import os
import pandas as pd

pd.set_option('display.max_colwidth', None)

# Load dataset
df = pd.read_csv('source_dataset.txt', names=['API', 'label'], sep='-SEP-')

# Check for null
print('Null lines:')
df.isnull().sum()

print('Dataset length:', len(df))

df.info(verbose=True)

# Drop null rows
df = df.dropna()

# Change label type to int
df.label = df.label.astype(int)

# Delete rows containing characters other than alphanumeric, whitespace and underscore
df = df[~df.API.str.contains(r'[^\w\s]')]

# Delete some repeated rows
df = df[df.API != 'lstrcpy InitCommonControls ']
df = df[df.API != '_CorExeMain ']
df = df[df.API != '_CorDllMain ']
df = df[df.API != 'LoadLibraryA ']

# Select only malicious files
df = df[df.label == 1]

# Drop label column
df.drop(columns=['label'], inplace=True)

# Change directory
os.chdir('../../')

# Save dataset
df.to_csv('source_final.txt', index=False, header=False)