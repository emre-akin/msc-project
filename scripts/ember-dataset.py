import os
import pandas as pd

# Change directory
os.chdir('/datasets/ember-dataset/ember2018/')
os.listdir()
features_path = []
features_path = os.listdir()

pd.set_option('display.max_colwidth', None)

# Create dictionary for features
def create_dict(dc):

    for i in dc:
        if not "features" in i:
            dc.remove(i)

    print(dc)
    print(len(dc))

    # /TODO DELETE THIS
    del dc[:5]
    del dc[1:]
    #

    print(dc)
    return dc

def data_process(x):

    for i in x:
        # Import Data set
        df = pd.read_json(i, lines=True)

        # Choose malicious files only
        df = df[df.label == 1]

        # Drop columns except function calls
        df.drop(df.columns[0:11], axis=1, inplace=True)
        df.drop(df.columns[[1,2]], axis=1, inplace=True)

        # Save dataframe to csv if not empty
        if not df.empty:
            df.to_csv('pre-train.csv', mode='a')


create_dict(features_path)
data_process(features_path)
print(features_path)

# Testing
'''
df = pd.read_json('train_features_0.jsonl', lines=True)
df.drop(df.columns[0:11], axis=1, inplace=True)
df.drop(df.columns[[1,2]], axis=1, inplace=True)
df.info(verbose=True)
df.head(1)
df.describe()
df[df.sha256 == 'ec86d25cbc434941f369e595ae3726f742ed4fa4c0627da65fa026a5e9fa1ccc'].label
len(df[df.label == 1])
df.columns
'''