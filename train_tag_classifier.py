# importing required libraries
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, GlobalAveragePooling1D, LSTM, Bidirectional, Dropout, \
    MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split

# loading small talk dataset
with open('data/small_talk.json') as file:
    data = json.load(file)

# creating list of utterances and corresponding labels
utterances = []
responses = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        utterances.append(pattern)
        labels.append(intent['tag'])

    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

class_size = len(list(set(labels)))
print('Number of distinctive classes: ', len(list(set(labels))))

# Transforming labels into model understandable form
label_encoder = LabelEncoder()
label_encoder.fit(labels)
labels = label_encoder.transform(labels)
y = tf.keras.utils.to_categorical(labels, class_size)

# Pre-processing data to load into model
vocab_size = 1000 # Limiting vocabulary size to 1000
max_len = 25 # Setting maximum length of sequences
oov_token = "<OOV>" # Deifning out of value tokens to cater for words outside the vocabulary during inference

# using tokenizer function to set the vocabulary size limit
tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, oov_token=oov_token)
tokenizer.fit_on_texts(utterances)
word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
print('Length of vocabulary: ', vocab_size)

sequences = tokenizer.texts_to_sequences(utterances)

# using max len to set all sequences to the same length
X = pad_sequences(sequences, truncating='post', maxlen=max_len)

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

# Cnn with LSTM Model Training
embedding_dimension = 64

model = Sequential()
model.add(Embedding(vocab_size, embedding_dimension, input_length=max_len))

model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5))

model.add(LSTM(64))
model.add(Dense(class_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

start = datetime.now()
history = model.fit(X, y, epochs=50, batch_size=20, validation_split=0.0)

model.save('models/classification/tag/tag_classfier.h5')
print('Runtime: ', datetime.now()-start)

model_accuracy = model.evaluate(x_test, y_test)
print('Test Set\n')
print('Loss: ', model_accuracy[0]*100)
print('Accuracy: ', model_accuracy[1]*100)
print('\n')

plt.title('Accuracy Plot')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

plt.title('Loss Plot')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Saving the fitted tokenizer as a pickle file
with open('models/classification/tag/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Saving the fitted label encoder as pickle file
with open('models/classification/tag/label_encoder.pickle', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file, protocol=pickle.HIGHEST_PROTOCOL)

''' Calling the models and pickled data
from keras.models import load_model
import numpy as np
def tag_classifier(user_text):
    max_len = 25
    model = load_model('models/classification/tag/tag_classfier.h5')
    
    with open('models/classification/tag/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    with open('models/classification/tag/label_encoder.pickle', 'rb') as encoder:
        label_encoder = pickle.load(encoder)
        
    pred = model.predict(pad_sequences(tokenizer.texts_to_sequences([user_text]), truncating='post', maxlen=max_len))
    tag = label_encoder.inverse_transform([np.argmax(pred)])
    
    for i in data['intents']:
        if i['tag'] == tag:
            print(np.random.choice(i['responses']))
'''