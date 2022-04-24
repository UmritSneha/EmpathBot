# Importing required libraries
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Concatenate, Dropout, Attention
from data_preprocessing import text_cleaning, clean_ans_ques, create_vocabulary, get_counsel_chat_data
import pickle

# Setting parameters
max_input_length = 20
emb_dim = 100


def get_counsel_chat_data():
    # Loading counsel chat dataset
    # fields = ['questionID', 'topic', 'questionTitle', 'answerText', 'answerSummary']
    counsel_chat_df = pd.read_csv('data/counsel_chat.csv', encoding='utf-8', low_memory=False)
    counsel_chat_df = counsel_chat_df[counsel_chat_df['topic'].isin(
        ['depression', 'anxiety', 'self-esteem', 'workplace-relationships', 'spirituality', 'sleep-improvement', 'grief-and-loss', 'substance-abuse',
         'eating-disorders', 'behavioral-change', 'addiction', 'legal-regulatory', 'professional-ethics', 'stress',
         'social-relationships', 'self-harm', 'diagnosis', 'counseling-fundamentals'])]
    print(counsel_chat_df.shape)
    questions = []
    answers = []

    for row in counsel_chat_df.iterrows():
        questions.append(row[1]['questionText'])
        answers.append(row[1]['answerSummary'])

    clean_ques, clean_ans = clean_ans_ques(questions, answers, 1000)
    # vocab = create_vocabulary(clean_ans, clean_ques)

    return clean_ques, clean_ans

# Get cleaned data
clean_ques, clean_ans = get_counsel_chat_data()

# Generate vocabulary
vocab = create_vocabulary(clean_ques, clean_ans)
vocab_size = len(vocab)
print('Length of vocabulary: ', vocab_size)

for i in range(len(clean_ans)):
    clean_ans[i] = '<SOS> ' + clean_ans[i] + ' <EOS>'

# get encoder input
encoder_input = []
for line in clean_ques:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])

    encoder_input.append(lst)
encoder_input = pad_sequences(encoder_input, max_input_length, padding='post', truncating='post')

# get decoder input
decoder_input = []
for line in clean_ans:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
    decoder_input.append(lst)
decoder_input = pad_sequences(decoder_input, max_input_length, padding= 'post', truncating= 'post')

# get decoder final output
decoder_final_output = []
for i in decoder_input:
    decoder_final_output.append(i[1:])

decoder_final_output = pad_sequences(decoder_final_output, max_input_length, padding='post', truncating='post')
decoder_final_output = to_categorical(decoder_final_output, vocab_size)
print('Final decoder output shape: ', decoder_final_output.shape)

# glove embedding
def glove_embedding():
    emb_index = {}
    with open(r'glove.6B/glove.6B.100d.txt', encoding='utf-8') as file:
        for line in file:
            items = line.split()
            word = items[0]
            numbers = np.asarray(items[1:], dtype='float32')
            emb_index[word] = numbers
    return emb_index

def embedding_matrix_creator(emb_dim, word_index):
    emb_matrix = np.zeros((len(word_index) + 1, emb_dim))
    emb_index = glove_embedding()

    for word, i in word_index.items():
        emb_vector = emb_index.get(word)
        if emb_vector is not None:
            emb_matrix[i] = emb_vector
    return emb_matrix
emb_matrix = embedding_matrix_creator(emb_dim, word_index=vocab)

enc_inp = Input(shape=(max_input_length, ))

embed = Embedding(vocab_size + 1, emb_dim, input_length=max_input_length, trainable=True)
embed.build((None,))
embed.set_weights([emb_matrix])
encoder_embed = embed(enc_inp)
encoder_lstm = Bidirectional(LSTM(400, return_state=True, dropout=0.08, return_sequences=True))
encoder_output, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embed)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

dec_inp = Input(shape=(max_input_length, ))
decoder_embed = embed(dec_inp)
decoder_lstm = LSTM(400 * 2, return_state=True, dropout=0.08, return_sequences=True)
decoder_output, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)

# attention
# attn_layer = tf.keras.layers.Attention()
# attn_op = attn_layer([encoder_output, decoder_output])
# decoder_concat_input = Concatenate(axis=-1)([decoder_output, attn_op])

dense = Dense(vocab_size, activation='softmax')
dense_output = dense(decoder_output)
# dense_output = dense(decoder_concat_input)


model = Model([enc_inp, dec_inp], dense_output)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
print(model.summary())

# foldername = 'models/response_generation/'
filepath = 'models/response_generation/' + 'seq2seq_model_cc_data.h5'
start = datetime.now()
history = model.fit([encoder_input, decoder_input], decoder_final_output, epochs=140, batch_size=50, validation_split=0.3)
model.save(filepath)
print('Runtime: ', datetime.now() - start)

plt.title('Accuracy Plot')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()

plt.title('Loss Plot')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# inference
# encoder model
encoder_model = Model(enc_inp, [encoder_output, encoder_states])

with open('models/response_generation/encoder_model_cc_data.json', 'w', encoding='utf8') as f:
    f.write(encoder_model.to_json())
encoder_model.save_weights('models/response_generation/encoder_model_weights_cc_data.h5')

# decoder model
decoder_state_input_h = Input(shape=(400 * 2,))
decoder_state_input_c = Input(shape=(400 * 2,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embed, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

decoder_model = Model([dec_inp] + decoder_states_inputs, [decoder_outputs] + decoder_states)
with open('models/response_generation/decoder_model_cc_data.json', 'w', encoding='utf8') as f:
    f.write(decoder_model.to_json())
decoder_model.save_weights('models/response_generation/decoder_model_weights_cc_data.h5')


# saving vocabulary as pickle file
with open('data/cc_vocab.pickle', 'wb') as handle:
    pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)


''''Calling models
def response_generation(user_input):

    max_input_length = 20

    # clean_ques, clean_ans = get_counsel_chat_data()
    # vocab = create_vocabulary(clean_ques, clean_ans)
    # inv_vocab = inverse_vocab(vocab)
    with open('data/cc_vocab.pickle', 'rb') as file:
        vocab = pickle.load(file)
    
    inv_vocab = {i:j for j, i in vocab.items()}

    # print(vocab)
    model = load_model('models/response_generation/seq2seq_model_cc_data.h5')
    encoder_model = load_mdl('models/response_generation/encoder_model_cc_data.json', 'models/response_generation/encoder_model_weights_cc_data.h5')
    decoder_model = load_mdl('models/response_generation/decoder_model_cc_data.json', 'models/response_generation/decoder_model_weights_cc_data.h5')
    # dense = model.layers[9]
    dense = model.layers[7]

    user_text = text_cleaning(user_input)
    user_text = [user_text]

    txt = []
    for i in user_text:
        lst = []
        for j in i.split():
            try:
                lst.append(vocab[j])
            except:
                lst.append(vocab['<OUT>'])
        txt.append(lst)

    txt = pad_sequences(txt, max_input_length, padding='post')

    enc_output, states_values = encoder_model.predict(txt)
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0,0] = vocab['<SOS>']
    stop_flag = False
    decoded_translation = ''

    while not stop_flag:
        dec_outputs, h, c = decoder_model.predict([empty_target_seq] + states_values)
        # print('Decoder output shape: ', dec_outputs.shape)
        
        # Attention
        # attn = model.layers[7]
        # attn_output = attn([enc_output, dec_outputs])
        # print('attention output shape: ', attn_output.shape)
        # decoder_concat_input = tf.keras.layers.Concatenate(axis=-1)([dec_outputs, attn_output])
        # decoder_concat_input = dense(decoder_concat_input)

        decoder_concat_input = dense(dec_outputs)

        sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])
        sampled_word = inv_vocab[sampled_word_index] + ' '
        # print(sampled_word)

        if sampled_word != '<EOS> ':
            decoded_translation += sampled_word

        if sampled_word == '<EOS> ' or len(decoded_translation.split()) > max_input_length:
            stop_flag = True

        empty_target_seq = np.zeros((1,1))
        empty_target_seq[0, 0] = sampled_word_index

        states_values = [h, c]
    print(decoded_translation)
'''