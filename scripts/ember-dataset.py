import os
import pandas as pd
import gc
import json

# Change directory
os.chdir('/datasets/ember-dataset/ember2018/')   # TODO change this
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

    # Save dataframe to csv if not empty
    #if not df.empty:
    #    df.to_csv('pre-train.csv', mode='a')

features_path = create_dict(features_path)
print(len(features_path))
print(features_path[2])
for i in features_path:
    data_process(features_path[i])

'''
# Testing
df = pd.read_json('train_features_5.jsonl', lines=True)
df.drop(df.columns[0:11], axis=1, inplace=True)
df.drop(df.columns[[1,2]], axis=1, inplace=True)
df.info(verbose=True)

len(df)

df = df.head(1000)
df.head(1)
df.describe()


test = df.to_json()
parsed = json.loads(test)
#print(type(parsed))
#json.dumps(parsed, indent=None)
data = []

with open('ember-test.txt', 'a') as f:
    for key, value in parsed.items():
        for value , value2 in value.items():
            for value2, value3 in value2.items():
                for item in value3:
                    _ = f.write('%s ' % item)
            _ = f.write('\n')
                #data[key] = value

del df
del test
del parsed
gc.collect()
'''