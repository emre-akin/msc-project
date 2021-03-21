import os
import pandas as pd
import gc
import json

# Change directory
os.chdir('datasets/ember-dataset/ember2018/')
os.listdir()
features_path = []
features_path = os.listdir()

pd.set_option('display.max_colwidth', None)

# Create dictionary for features
def create_dict(dc):

    return [i for i in dc if 'features' in i]

def data_process(i):

    print('Processing ' + i)
    
    # Import Data set
    df = pd.read_json(i, lines=True)

    # Choose malicious files only
    df = df[df.label == 1]

    # Drop columns except function calls
    df.drop(df.columns[0:11], axis=1, inplace=True)
    df.drop(df.columns[[1,2]], axis=1, inplace=True)

    # Dataframe to json
    df_json = df.to_json()

    # Parse dataframe
    df_parsed = json.loads(df_json)

    # Write to pre-train.txt
    with open('pre-train.txt', 'a') as f:
        for key, value in df_parsed.items():
            for value , value_2 in value.items():
                for value_2, value_3 in value_2.items():
                    for item in value_3:
                        _ = f.write('%s ' % item)
                _ = f.write('\n')

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
