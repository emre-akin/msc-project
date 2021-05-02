###########################################################################################################
###  EMBER dataset reference
#    Title: EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models
#    Author: Hyrum S. Anderson, Phil Roth
#    Date: 2018
#    Availability: https://github.com/elastic/ember
###########################################################################################################

import os
import pandas as pd
import gc
import json

features_path = []
features_path = os.listdir()

pd.set_option('display.max_colwidth', None)

# Create dictionary for features
def create_dict(dc):

    return [i for i in dc if 'features' in i]

def data_process(i):

    print('Processing ', i)
    
    # Import Data set
    df = pd.read_json(i, lines=True)

    # Drop unlabeled data
    df.drop(df[df['label'] == -1].index, inplace=True)

    # Drop columns except function calls and labels
    df.drop(df.columns[0:3], axis=1, inplace=True)
    df.drop(df.columns[1:8], axis=1, inplace=True)
    df.drop(df.columns[2:], axis=1, inplace=True)

    # Dataframe to json
    df_json = df.to_json()

    # Parse dataframe
    df_parsed = json.loads(df_json)

    # Write to source_dataset.txt
    with open('source_dataset.txt', 'a') as f:

        imports = df_parsed['imports']
        label = df_parsed['label']

        label_iter = iter(list(label.values()))
        for value , value_2 in imports.items():
            exist = False
            for value_2, value_3 in value_2.items():
                for item in value_3:
                    exist = True
                    _ = f.write('%s ' % item)
            if exist:
                _ = f.write('-SEP- %d \n' %next(label_iter))

    
    # Garbage collection
    #input('test')
    del df
    df = None
    del df_json
    df_json = None
    del df_parsed
    df_parsed = None
    gc.collect()


features_path = create_dict(features_path)

for i in features_path:
    data_process(features_path[i])
