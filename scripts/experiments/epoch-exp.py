###########################################################################################################
###  Tensorflow reference
#    Title: Text classification with an RNN
#    Author: Google, Tensorflow
#    Date: 2021
#    Availability: https://www.tensorflow.org/tutorials/text/text_classification_rnn 
###########################################################################################################
### Transfer learning reference
#    Title: Transfer learning and fine-tuning
#    Author: Google, Tensorflow
#    Date: 2021
#    Availability: https://www.tensorflow.org/tutorials/images/transfer_learning
###########################################################################################################
### Preprocessing, LSTM and embedding reference
#    Title: Classify Toxic Online Comments with LSTM and GloVe
#    Author: Susan Li
#    Date: 2019
#    Availability: https://github.com/susanli2016/NLP-with-Python/blob/master/Toxic%20Comments%20LSTM%20GloVe.ipynb
###########################################################################################################
###  Mal-API-2019 dataset reference
#    Title: Deep learning based Sequential model for malware analysis using Windows exe API Calls
#    Author: Ahmet Faruk Yazı, Ferhat Ozgur Catak, Ensar Gul
#    Date: 2020
#    Availability: https://github.com/ocatak/malware_api_class
###########################################################################################################

# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Define plot
import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

# Define f1 score
import keras.backend as K

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

### Preprocessing

# Load target dataset & tokenize
df = pd.read_csv('/datasets/target_final.txt', names=['calls', 'types', 'Backdoor', 'Trojan', 'Spyware', 'Worms', 'Dropper', 'Downloader', 'Virus', 'Adware'])

calls_list = []
for i in range(len(df)):
  calls_list.append(df.calls[i])

labels = ['Backdoor', 'Trojan', 'Spyware', 'Worms', 'Dropper', 'Downloader', 'Virus', 'Adware']
y = df[labels].values

tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(calls_list)
sequences = tokenizer.texts_to_sequences(calls_list)
word_index = tokenizer.word_index
print('Vocabulary size:', len(word_index))

data = pad_sequences(sequences, padding = 'post', maxlen = 1000)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)

indices = np.arange(data.shape[0])
np.random.seed(0)
np.random.shuffle(indices)
data = data[indices]
labels = y[indices]

num_validation_samples = int(0.2*data.shape[0])
x_train = data[: -num_validation_samples]
y_train = labels[: -num_validation_samples]
x_val = data[-num_validation_samples: ]
y_val = labels[-num_validation_samples: ]
print('Number of entries in each category:')
print('training: ', y_train.sum(axis=0))
print('validation: ', y_val.sum(axis=0))

### Training

# Load GloVe word embedding
embeddings_index = {}
embedding_path = "/embeddings/target_glove.txt"
f = open(embedding_path)
print('Loading GloVe from:', embedding_path, '...', end='')
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()
print("Done.\n Proceeding with Embedding Matrix...", end="")
np.random.seed(10)
embedding_matrix = np.random.random((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(" Completed!")

# Configure the model
sequence_input = Input(shape=(1000,), dtype='int32')
embedding_layer = Embedding(len(word_index) + 1,
                           100,
                           weights = [embedding_matrix],
                           input_length = 1000,
                           trainable=False,
                           name = 'embeddings')
embedded_sequences = embedding_layer(sequence_input)
x = Bidirectional(LSTM(256, return_sequences=True,name='lstm_layer'))(embedded_sequences)
x = GlobalMaxPool1D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
preds = Dense(8, activation="softmax")(x)

# Compile the model
model = Model(sequence_input, preds)
model.compile(loss = 'binary_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])

# Train the model
print('Training progress:')
history = model.fit(x_train, y_train, epochs = 200, batch_size=64, validation_data=(x_val, y_val))

# Plot loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show();
plt.savefig('/models/200epoch/target_loss_200epoch.png')

# Plot accuracy
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.plot(epochs, accuracy, label='Training accuracy')
plt.plot(epochs, val_accuracy, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
#plt.show();
plt.savefig('/models/200epoch/target_accuracy_200epoch.png')