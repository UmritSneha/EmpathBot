# Importing libraries
import json
import os

import pandas as pd
import numpy as np
import re, nltk

from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import tensorflow
import pickle
from nltk.corpus import stopwords
from keras.models import model_from_json

stop_words = set(stopwords.words("english"))

def remove_null_rows():
    csv_file = pd.read_csv('data/counsel_chat.csv', encoding='utf-8', low_memory=False)
    csv_file = csv_file.dropna(how='all')
    csv_file.to_csv('data/counsel_chat.csv')
    print('Null rows removed from counsel chat dataset')

# remove_null_rows()

def pickle(filepath):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        filename = os.path.splitext(filepath)[0] + '.pkl'
        df.to_pickle(filename)

# pickle("data/counsel_chat-old.csv")

def unpickle_csv(filepath):
    df = pd.read_pickle('filepath')
    filename = os.path.splitext(filepath)[0] + '.csv'
    df.to_csv(filename)

# def unpickle_text(filepath):

def capitalize(text):
    line_list = text.split('.')
    new_text = []

    for val in line_list:
        val = val.strip()
        new_text += val.capitalize()+'.'

    new_text = new_text[:-2]

    return new_text

def text_cleaning(text):
    replace_brackets = re.compile('[/(){}\|[\]]')
    replace_symbols = re.compile('[^0-9a-z #+_@,;]')

    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"dont", "do not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"_comma_", " ", text)
    text = replace_brackets.sub(' ', text)
    text = replace_symbols.sub('', text)

    # text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Tokenising and removing punctuation
tokenizer = nltk.RegexpTokenizer(r"\w+")
def text_tokenizer(texts):
    tokens = []
    for text in texts:
        tokens.append(tokenizer.tokenize(text))
    return tokens

# Removing stopwords and normalising casing
english_stopwords = stopwords.words('English')
def stopword_removal(tokens):
    docs = []
    for token in tokens:
        docs.append([word.lower() for word in token if word not in english_stopwords])
    return docs

# Performing stemming
sb_stemmer = SnowballStemmer('english')
def document_stemmer(docs):
    stemmed_docs = []
    for doc in docs:
        stemmed_docs.append([sb_stemmer.stem(word) for word in doc])
    return stemmed_docs

def tokenize(data):
    t = Tokenizer(num_words=200, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    t = t.fit_on_texts(data.values)

    return t

def tokenize_and_pad(features, labels, max_input_length, vocab_size):
    
    tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(features.values)
    X = tokenizer.texts_to_sequences(features.values)
    X = pad_sequences(X, maxlen=max_input_length)
    # print('Shape of data tensor: ', X.shape)


    Y = pd.get_dummies(labels).values
    # print('Shape of label tensor: ', Y.shape)

    labels = list(sorted(set(labels)))

    return X,Y, tokenizer, labels

def load_emotion_classifier_data(max_input_length, vocab_size):

    ed_train_df = ed_data_loading()

    ed_train_df['prompt'] = ed_train_df['prompt'].apply(text_cleaning)
    # ed_test_df['prompt'] = ed_test_df['prompt'].apply(text_cleaning)

    X, Y, tokenizer, labels = tokenize_and_pad(ed_train_df['prompt'], ed_train_df['context'], max_input_length, vocab_size)
    # X_test, Y_test = tokenize_and_pad(ed_test_df['prompt'], ed_test_df['context'], max_input_length, vocab_size)

    return X, Y, tokenizer, labels

def create_intent_classifier_data():
    ## text classification
    label_dir = {
        "small talk": "data/small_talk.json",
        "counsel chat": "data/counsel_chat - Copy.csv"
    }

    data = []
    labels = []

    for label in label_dir:
        filepath = label_dir[label]
        if filepath.endswith(".csv"):
            csv_file = pd.read_csv(filepath)
            for row in csv_file.iterrows():
                data.append(row[1]['questionTitle'])
                labels.append(label)

        elif filepath.endswith('.json'):
            json_file = json.load(open(filepath, 'r'))
            for intent in json_file['intents']:
                for pattern in intent['patterns']:
                    data.append(pattern)
                    labels.append(label)

    df = pd.DataFrame(data, columns=['user_text'])
    df['labels'] = labels
    df.to_csv('data/user_intent.csv')
    print('Intent classifier data created')

# create_intent_classifier_data()


def load_intent_classifier_data(max_input_length, vocab_size):
    df = pd.read_csv('data/user_intent.csv')
    df['user_text'] = df['user_text'].apply(text_cleaning)

    tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['user_text'].values)
    X = tokenizer.texts_to_sequences(df['user_text'].values)
    X = pad_sequences(X, maxlen=max_input_length)
    # print('Shape of data tensor:', X.shape)

    Y = pd.get_dummies(df['labels']).values
    # print('Shape of label tensor:', Y.shape)

    return X, Y, tokenizer

# generate question and answer set to be used for deep learning model
def clean_ans_ques(questions, answers, max_input_length):

    # perform text cleaning
    sorted_ques = []
    sorted_ans = []
    for i in range(len(questions)):
        if len(questions[i]) < max_input_length:
            sorted_ques.append(questions[i])
            sorted_ans.append(answers[i])

    clean_ques = []
    clean_ans = []
    for line in sorted_ques:
        clean_ques.append(text_cleaning(line))

    for line in sorted_ans:
        clean_ans.append(text_cleaning(line))

    #for i in range(len(clean_ans)):
    #    clean_ans[i] = ' '.join(clean_ans[i].split()[:30])
    return clean_ques, clean_ans


def create_vocabulary(clean_ans, clean_ques):
    word2count = {}

    for line in clean_ques:
        for word in line.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    for line in clean_ans:
        for word in line.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    threshold = 5
    vocab = {}
    word_num = 0

    for word, count in word2count.items():
        if count >= threshold:
            vocab[word] = word_num
            word_num += 1

    del(word2count, word, count, threshold, word_num)
    tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
    x = 0
    for token in tokens:
        vocab[token] = x
        x += 1

    return vocab

def inverse_vocab(vocab):
    inv_vocab = {i:j for j, i in vocab.items()}
    return inv_vocab

def ed_data_loading():
    fields = ['conv_id', 'context', 'prompt', 'utterance']

    ed_train_df = pd.read_csv('data/empathetic_dialogues_train.csv', encoding='utf-8', low_memory=False, usecols=fields)
    ed_train_df['responseLength'] = ed_train_df['utterance'].apply(lambda x: len(str(x).split(' ')))
    ed_train_df.drop(ed_train_df[ed_train_df.responseLength > 80].index, inplace=True)

    # ed_test_df = pd.read_csv('data/empathetic_dialogues_test.csv', encoding='utf-8', low_memory=False, usecols=fields)
    # ed_test_df['responseLength'] = ed_test_df['utterance'].apply(lambda x: len(str(x).split(' ')))
    # ed_test_df.drop(ed_test_df[ed_test_df.responseLength > 80].index, inplace=True)

    # ed_valid_df = pd.read_csv('data/empathetic_dialogues_valid.csv', encoding='utf-8', low_memory=False, usecols=fields)
    # ed_valid_df['responseLength'] = ed_valid_df['utterance'].apply(lambda x: len(str(x).split(' ')))
    # ed_valid_df.drop(ed_valid_df[ed_valid_df.responseLength > 80].index, inplace=True)

    return ed_train_df


def get_ed_data():
    fields = ['conv_id', 'context', 'prompt', 'utterance']

    ed_train_df = pd.read_csv('data/empathetic_dialogues_train.csv', encoding='utf-8', low_memory=False, usecols=fields)
    ed_train_df['responseLength'] = ed_train_df['utterance'].apply(lambda x: len(str(x).split(' ')))
    ed_train_df.drop(ed_train_df[ed_train_df.responseLength > 80].index, inplace=True)

    questions = []
    answers = []

    for row in ed_train_df.iterrows():
        questions.append(row[1]['prompt'])
        answers.append(row[1]['utterance'])

    clean_ques, clean_ans = clean_ans_ques(questions, answers, 200)
    return clean_ques, clean_ans

def get_counsel_chat_data():
    # Loading counsel chat dataset
    # fields = ['questionID', 'topic', 'questionTitle', 'answerText', 'answerSummary']
    counsel_chat_df = pd.read_csv('data/counsel_chat.csv', encoding='utf-8', low_memory=False)
    counsel_chat_df = counsel_chat_df[counsel_chat_df['topic'].isin(['depression', 'anxiety', 'self-esteem', 'workplace-relationships', 'spirituality', 'trauma', 'anger-management', 'sleep-improvement', 'grief-and-loss', 'substance-abuse', 'family-conflict', 'eating-disorders', 'behavioral-change', 'addiction', 'legal-regulatory', 'professional-ethics', 'stress', 'social-relationships', 'self-harm', 'diagnosis', 'counseling-fundamentals'])]

    questions = []
    answers = []

    for row in counsel_chat_df.iterrows():
        questions.append(row[1]['questionText'])
        answers.append(row[1]['answerText'])

    clean_ques, clean_ans = clean_ans_ques(questions, answers, 1000)
    # vocab = create_vocabulary(clean_ans, clean_ques)

    return clean_ques, clean_ans

def load_mdl(model_filename, model_weights_filename):
    with open(model_filename, 'r', encoding='utf8') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights_filename)
    return model

def generate_intent_classifier_data():
    label_dir = {
        "small talk": "data/small_talk.json",
        "counsel chat": "data/counsel_chat.csv"
    }

    data = []
    labels = []

    for label in label_dir:
        filepath = label_dir[label]
        if filepath.endswith(".csv"):
            csv_file = pd.read_csv(filepath)
            for row in csv_file.iterrows():
                data.append(row[1]['questionTitle'])
                labels.append(label)

        elif filepath.endswith('.json'):
            json_file = json.load(open(filepath, 'r'))
            for intent in json_file['intents']:
                for pattern in intent['patterns']:
                    data.append(pattern)
                    labels.append(label)

    df = pd.DataFrame(data, columns=['user_text'])
    df['labels'] = labels
    df.to_csv('data/user_intent.csv')

    print('Intent classifier data generated!')


# creating a list of questions and answers
def generate_question_answer_sample():
    questions = []
    answers = []

    # Loading small talk dataset
    st_data = json.load(open('data/small_talk.json', 'r'))

    # Loading counsel chat dataset
    fields = ['questionID', 'topic', 'questionTitle', 'answerSummary']
    counsel_chat_df = pd.read_csv('data/counsel_chat.csv', encoding='utf-8', low_memory=False, usecols=fields)

    for row in counsel_chat_df.iterrows():
        questions.append(row[1]['questionTitle'])
        answers.append(row[1]['answerSummary'])

    for intent in st_data['intents']:
        for pattern in intent['patterns']:
            for response in intent['responses']:
                questions.append(pattern)
                answers.append(response)


    df = pd.DataFrame(questions, columns=['questions'])
    df['answers'] = answers
    df.to_csv('data/evaluation_data.csv')
    print('Evaluation data sample has been generated!')
generate_question_answer_sample()
