###########################################################################################################
###  Mittens reference
#    Title: Fine tune GloVe embeddings using Mittens
#    Author: Sivasurya Santhanam
#    Date: 2020
#    Availability: https://towardsdatascience.com/fine-tune-glove-embeddings-using-mittens-89b5f3fe4c39
###########################################################################################################
###  EMBER dataset reference
#    Title: EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models
#    Author: Hyrum S. Anderson, Phil Roth
#    Date: 2018
#    Availability: https://github.com/elastic/ember
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
from mittens import Mittens
import csv

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

# Load GloVe model
def glove2dict(glove_filename):
    with open(glove_filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ',quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:-1])))
                for line in reader}
    return embed
glove_path = "/embeddings/source_glove.txt"
pre_glove = glove2dict(glove_path)

# Create & train mittens model
mittens_model = Mittens(n=100, max_iter=100)
new_embeddings = mittens_model.fit(
    corpus.matrix.A,
    vocab=corpus.dictionary,
    initial_embedding_dict= pre_glove)

# Save the word embedding
with open("/embeddings/tf_finetuned_glove.txt", "w") as f:
    for word in corpus.dictionary:
        f.write(word)
        f.write(" ")
        for i in range(0, 100):
            f.write(str(new_embeddings[corpus.dictionary[word]][i]))
            f.write(" ")
        f.write("\n")