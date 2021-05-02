###########################################################################################################
###  GloVe model reference
#    Title: Word vectorization using GloVe
#    Author: Japneet Singh Chawla
#    Date: 2018
#    Availability: https://medium.com/analytics-vidhya/word-vectorization-using-glove-76919685ee0b
###########################################################################################################
###  Mal-API-2019 dataset reference
#    Title: Deep learning based Sequential model for malware analysis using Windows exe API Calls
#    Author: Ahmet Faruk Yazı, Ferhat Ozgur Catak, Ensar Gul
#    Date: 2020
#    Availability: https://github.com/ocatak/malware_api_class
###########################################################################################################

# Import libraries
import pandas as pd
from glove import Corpus, Glove

# Load datasets
train = pd.read_csv('/datasets/api-call-dataset/target_train_numerized.txt', names=['calls', 'types'])
test = pd.read_csv('/datasets/api-call-dataset/target_test_numerized.txt', names=['calls', 'types'])

# Combine datasets
df = train.append(test, ignore_index=True)

# Delete older datasets to free memory
del train
del test

# Get words
new_lines = [line.split() for line in df.calls]

# Create new corpus
corpus = Corpus() 

corpus.fit(new_lines, window=10)

# Create GloVe model
glove = Glove(no_components=100, learning_rate=0.05)

# Train the GloVe model
glove.fit(corpus.matrix, epochs=100, no_threads=4, verbose=True)

# Add corpus dictionary to GloVe model
glove.add_dictionary(corpus.dictionary)

# Save the GloVe model
with open("/embeddings/target_glove.txt", "w") as f:
    for word in glove.dictionary:
        f.write(word)
        f.write(" ")
        for i in range(0, 100):
            f.write(str(glove.word_vectors[glove.dictionary[word]][i]))
            f.write(" ")
        f.write("\n")