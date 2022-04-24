import random
import urllib.parse
from datetime import datetime

import pandas as pd
from keras.models import load_model
from data_preprocessing import text_cleaning, load_mdl
from evaluation_metric import  question_similarity
import numpy as np
from text_similarity import cosine_similarity
import json, pickle
import pickle5 as pickle

from keras_preprocessing.sequence import pad_sequences
from database_handler import DatabaseManager
import googlesearch # used for helpline retrieval
import aiml
NoneType = type(None)
import warnings
warnings.filterwarnings("ignore")

kernel = aiml.Kernel()
kernel.learn("std-startup.xml")
kernel.respond("load aiml b")

class ChatBot:

    def __init__(self, name):
        self.name = name

    def response_retrieval(self, user_input):
        # load small talk dataset
        with open('data/small_talk.json') as file:
            data = json.load(file)

        max_len = 25
        model = load_model('models/classification/tag/tag_classfier.h5')

        with open('models/classification/tag/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        with open('models/classification/tag/label_encoder.pickle', 'rb') as encoder:
            label_encoder = pickle.load(encoder)

        pred = model.predict(
            pad_sequences(tokenizer.texts_to_sequences([user_input]), truncating='post', maxlen=max_len))
        tag = label_encoder.inverse_transform([np.argmax(pred)])

        for i in data['intents']:
            if i['tag'] == tag[0]:
                # print('Tag: ', tag[0])
                response = np.random.choice(i['responses'])
        return response, tag[0]

    def counsel_chat_response_shuffling(self, question, df):
        df = df.loc[df['questionTitle'] == question]
        # print(df)
        df = df.sample()
        for row in df.iterrows():
            ans = row[1]['answerSummary']
        return ans

    def response_generation(self, user_input):

        max_input_length = 20

        with open('data/cc_vocab.pickle', 'rb') as file:
            vocab = pickle.load(file)

        inv_vocab = {i: j for j, i in vocab.items()}

        # print(vocab)
        model = load_model('models/response_generation/seq2seq_model_cc_data.h5')
        encoder_model = load_mdl('models/response_generation/encoder_model_cc_data.json',
                                 'models/response_generation/encoder_model_weights_cc_data.h5')
        decoder_model = load_mdl('models/response_generation/decoder_model_cc_data.json',
                                 'models/response_generation/decoder_model_weights_cc_data.h5')
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
        empty_target_seq[0, 0] = vocab['<SOS>']
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

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index

            states_values = [h, c]

        return decoded_translation

    def emotion_classifier(self, user_input):
        max_len = 25
        model = load_model('models/classification/emotion/emotion_classifier.h5')

        with open('models/classification/emotion/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        with open('models/classification/emotion/label_encoder.pickle', 'rb') as encoder:
            label_encoder = pickle.load(encoder)

        pred = model.predict(
        pad_sequences(tokenizer.texts_to_sequences([user_input]), truncating='post', maxlen=max_len))
        emotion = label_encoder.inverse_transform([np.argmax(pred)])

        return emotion[0]

    def intent_classifier(self, user_input):
        max_len = 30
        model = load_model('models/classification/intent/intent_classfier.h5')

        with open('models/classification/intent/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        with open('models/classification/intent/label_encoder.pickle', 'rb') as encoder:
            label_encoder = pickle.load(encoder)

        pred = model.predict(
            pad_sequences(tokenizer.texts_to_sequences([user_input]), truncating='post', maxlen=max_len))
        label = label_encoder.inverse_transform([np.argmax(pred)])
        return label[0]

    # check confidence of bot response
    # if confidence of response is less than 0.2 detect user emotion and use
    def check_confidence(self, response, user_text):
        df = pd.read_csv('data/evaluation_data.csv')
        questions = []
        answers = []
        for row in df.iterrows():
            questions.append(row[1]['questions'])
            answers.append(row[1]['answers'])

        sample = question_similarity(user_text, questions)
        indices = [i for i, x in enumerate(questions) if x == sample]
        references = []
        scores = []
        for idx in indices:
            references.append(answers[idx])

        for each in references:
            # print(each)
            # print(response)
            scores.append(cosine_similarity(response, each))

        return max(scores)

    def timely_greeting(self):
        currentTime = datetime.datetime.now()
        if currentTime.hour > 12:
            greeting = ['Good morning <USER_FIRSTNAME>', 'Morning <USER_FIRSTNAME>']
        elif 12 <= currentTime.hour < 18:
            greeting = ['Good afternoon <USER_FIRSTNAME>', 'Afternoon <USER_FIRSTNAME>']
        else:
            greeting = ['Good evening <USER_FIRSTNAME>']
        return greeting

    def get_helpline(self):
        '''g = geocoder.ip('me')
        coordinates = g.latlng
        result = rg.search(coordinates)
        # print(result)

        for each in result:
            location = each.get('name')
            # print(location)'''

        query = "mental health helpline"
        for url in googlesearch.search(query):
            link = urllib.parse.unquote(url)
        return link
# Prompting user to enter their first name and email
# Verifies user and stores into database
# user_first_name = input('Name: ')
# user_email = input('Email: ')

# creating a database connection
db_manager = DatabaseManager('EmpathBot.db')
db_manager.check_database()
db_manager.create_user_table()
db_manager.create_user_feedback_table()
db_manager.create_chat_history_table()
# db_manager.insert_user_table(user_first_name, user_email)
# db_manager.insert_user_table(user_first_name, user_email)


'''def initiate_conversation():
    actives = db_manager.fetch_active()
    if actives > 1:
        response = random.choice(
            ["Welcome back <USER_FIRSTNAME>, it's so nice to see you!",
             "Welcome back <USER_FIRSTNAME>, how's your day going so far?"])
    else:
        response = random.choice(["Hello <USER_FIRSTNAME>! I'm Zoey, an AI empathetic bot.",
                                  "Hi <USER_FIRSTNAME> I'm Zoey, an AI empathetic bot."])
    response = response.replace('<USER_FIRSTNAME>', db_manager.fetch_user_name())

    return response'''
'''chatbot = ChatBot('EmpathBot')
answer = 'yes sure <HELPLINE>'
r = chatbot.get_helpline()
print(answer.replace('<HELPLINE>', str(r)))'''


def get_final_response(user_input):
    ## prompt user to enter text
    # user_input = input('User: ')
    apology_responses_1 = ["<USER_FIRSTNAME>, I understand you feel <USER_EMOTION>. Please go on...", "I see you feel <USER_EMOTION>, <USER_FIRSTNAME>. Can you elaborate on that?"]
    apology_responses_2 = ["I'm sorry, I don't quite understand what you mean by <USER_TEXT>", "I'm sorry, but what do you mean by <USER_TEXT>?", "<USER_FIRSTNAME>, you said <USER_TEXT>! What do you mean by that?", "<USER_FIRSTNAME>, I understand you are feeling <USER EMOTION>. But, can you please elaborate on <USER_TEXT>"]

    chatbot = ChatBot('EmpathBot')

    # determine user intent
    user_intent = chatbot.intent_classifier(user_input)
    # print('User intent: ', user_intent)

    # determine user emotion
    user_emotion = chatbot.emotion_classifier(user_input)
    # print('User emotion: ', user_emotion)

    if user_intent == "small talk":
        response, tag = chatbot.response_retrieval(user_input)

    elif user_intent == "counsel chat":
        response = chatbot.response_generation(user_input)
        tag = "NewResponse"
        response = response.capitalize()

        # check confidence of response
        response_confidence = chatbot.check_confidence(response, user_input)
        print('Response confidence: ', response_confidence)
        # if confidence is lower than 0.2
        if response_confidence < 0.5 and response_confidence > 0.0:
            try:
                response = kernel.respond(user_input)
                if len(response) == 0:
                    response = random.choice(apology_responses_1)
            except RuntimeWarning:
                response = random.choice(apology_responses_1)

        elif response_confidence == 0:
            response = random.choice(apology_responses_2)

    ## check if response contains tags and replace tags
    if '<USER_FIRSTNAME>' in response:
        response = response.replace('<USER_FIRSTNAME>', db_manager.fetch_user_name())
        
    if '<USER_EMOTION>' in response:
        response = response.replace('<USER_EMOTION>', user_emotion)

    if '<PAD>' in response:
        response = response.replace('<PAD>', '')
    if '<OUT>' in response:
        response = response.replace('<OUT>', '')
    if '<SOS>' in response:
        response = response.replace('<SOS>', 'you')
    if '<HELPLINE>' in response:
        helpline = chatbot.get_helpline()
        response = response.replace('<HELPLINE>', helpline)
        # print('response: ', response)
    if 'hi new york' in response:
        response = response.replace('hi new york', '')
    if '<USER_TEXT>' in response:
        response = response.replace('<USER_TEXT>', user_input)

    if db_manager.fetch_previous_tag() == "UserEncouragement":
        response = random.choice(["Alright, Would you like to try out some mindfulness activities?", "Great, Would you like to try some mindfulness activities?", "Nice, Would you like to do some mindfulness activities with me?"])

    if db_manager.fetch_previous_tag() == "jokes":
        response = "Is there anything else you would like to hear?"

    db_manager.insert_chat_table(user_input, response, user_intent, user_emotion, tag)
    # display response
    return response

'''exit_bot = False
user_exits = ['exit', 'quit']
bot_exits = ['Take care! It was nice talking to you.', 'Bye, take care.', 'See you later. I hope I was of help to you.']
bot_text = initiate_conversation()
print('Bot: ', bot_text)
chatbot = ChatBot('EmpathBot')

while not exit_bot:
    user_input = input('User: ' )
    if user_input.lower() in user_exits:
        exit_bot = True
        print('Bot: ', random.choice(bot_exits))
    else:
        print('Bot: ', get_final_response(user_input))'''

# closing connection
# db_manager.close_connection()
