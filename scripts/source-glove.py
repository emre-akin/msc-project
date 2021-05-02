###########################################################################################################
###  GloVe model reference
#    Title: Word vectorization using GloVe
#    Author: Japneet Singh Chawla
#    Date: 2018
#    Availability: https://medium.com/analytics-vidhya/word-vectorization-using-glove-76919685ee0b
###########################################################################################################
###  EMBER dataset reference
#    Title: EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models
#    Author: Hyrum S. Anderson, Phil Roth
#    Date: 2018
#    Availability: https://github.com/elastic/ember
###########################################################################################################

# Import libraries
import pandas as pd
from glove import Corpus, Glove

# Load the dataset
df = pd.read_csv('/datasets/source_final.txt', names=['calls'])

df.head(3)

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
with open("/embeddings/source_glove.txt", "w") as f:
    for word in glove.dictionary:
        f.write(word)
        f.write(" ")
        for i in range(0, 100):
            f.write(str(glove.word_vectors[glove.dictionary[word]][i]))
            f.write(" ")
        f.write("\n")