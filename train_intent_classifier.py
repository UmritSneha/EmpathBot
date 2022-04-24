# importing required libraries
import json
import numpy as np
import matplotlib.pyplot as plt
# importing required libraries
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, GlobalAveragePooling1D, LSTM, Bidirectional, Dropout, \
    MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
from keras.regularizers import l1

from data_preprocessing import text_cleaning, generate_intent_classifier_data
# generate new dataset
# generate_intent_classifier_data()

# Loading the dataset
utterances = []
intents = []
df = pd.read_csv('data/user_intent.csv', usecols=['user_text', 'labels'])
df = df.drop_duplicates(subset=['user_text'])
df['user_text'] = df['user_text'].apply(text_cleaning)

for row in df.iterrows():
    utterances.append(row[1]['user_text'])
    intents.append(row[1]['labels'])

class_size = len(list(set(intents)))
print('Number of distinctive classes: ', len(list(set(intents))))

# Transforming labels into model understandable form
label_encoder = LabelEncoder()
label_encoder.fit(intents)
intents = label_encoder.transform(intents)
y = tf.keras.utils.to_categorical(intents, num_classes=2)

# Pre-processing data to load into model
max_len = 30
oov_token = "<OOV>" # Defining out of vocab tokens to cater for words outside the vocabulary during inference

# using tokenizer function to set the vocabulary size limit
tokenizer = Tokenizer(num_words=50000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, oov_token=oov_token)
tokenizer.fit_on_texts(utterances)
word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
print('Length of vocabulary: ', vocab_size)

sequences = tokenizer.texts_to_sequences(utterances)

# using max len to set all sequences to the same length
X = pad_sequences(sequences, truncating='post', maxlen=max_len)
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

# LSTM
embedding_dimension = 16
model = Sequential()
model.add(Embedding(vocab_size, embedding_dimension, input_length=max_len))
# model.add(SpatialDropout1D(0.5))
# model.add(LSTM(64))
model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5))

model.add(Dense(class_size, activation='softmax', activity_regularizer=l1(0.001)))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())

'''# Cnn with LSTM Model Training
embedding_dimension = 64

model = Sequential()
model.add(Embedding(vocab_size, embedding_dimension, input_length=max_len))

model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5))

model.add(LSTM(64, dropout=0.08, recurrent_dropout=0.08))
model.add(Dense(class_size, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())
'''

## start training
start = datetime.now()
history = model.fit(x_train, y_train, epochs=10, batch_size=5, validation_split=0.3, verbose=0)
model.save('models/classification/intent/intent_classfier.h5')
print('Runtime: ', datetime.now()-start)

model_accuracy = model.evaluate(x_test, y_test)
print('Test Set\n')
print('Loss: ', model_accuracy[0]*100)
print('Accuracy: ', model_accuracy[1]*100)
print('\n')

plt.title('Accuracy Plot')
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show()

plt.title('Loss Plot')
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

# Saving the fitted tokenizer as a pickle file
with open('models/classification/intent/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Saving the fitted label encoder as pickle file
with open('models/classification/intent/label_encoder.pickle', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file, protocol=pickle.HIGHEST_PROTOCOL)


'''Calling models
def intent_classifier(user_text):
    max_len = 25
    model = keras.models.load_model('models/classification/intent/intent_classfier.h5')
    
    with open('models/classification/intent/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    with open('models/classification/intent/label_encoder.pickle', 'rb') as encoder:
        label_encoder = pickle.load(encoder)
        
    pred = model.predict(pad_sequences(tokenizer.texts_to_sequences([user_text]), truncating='post', maxlen=max_len))
    label = label_encoder.inverse_transform([np.argmax(pred)])
    
    print(label)
'''
