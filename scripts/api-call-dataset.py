import sys
from zipfile import PyZipFile
import pandas as pd

# Unzip referance: https://docs.python.org/3/library/zipfile.html
for zipLocation in sys.argv[1:]:
    pzf = PyZipFile(zipLocation)
    pzf.extractall()

# Import data set
df = pd.read_csv("all_analysis_data.txt", names=["API Calls"])
# Import labels
labels = pd.read_csv("labels.csv", names=["Malware Type"])

len(df)
len(labels)

'''
df.info(verbose=True)
df.head(1)
'''

# Combine dataframes
df = df.join(labels)

'''
df.head(10)
df.describe()
df.iloc[[0],[0]]
'''

# Create train and test data sets / 90% - 10%
train = df[:6397]
test = df[6397:]

len(train)
len(test)

# Save train and test data sets
train.to_csv("train.csv", index=None)
test.to_csv("test.csv", index=None)