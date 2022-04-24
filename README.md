# ZOEY, THE EMPATHETIC BOT
This file provides the instructions required to run the implemented conversational agent along with a brief overview of the different components of the implementation codes.

## Aim and Objectives
The aim of this project is to address the lack of empathy and specificity found in several existing chatbots by implementing a text-based bot that can understand user emotions and respond convincingly to express its empathetic support in ways that approaches or matches human capabilities.The key objectives to achieve this work are as follows:
1. Research for an appropriate corpus that focuses on one-to-one emotional support conversations, preferably derived from a counselling domain with a verified speaker.
2. Pre-process raw data using normalization and perform exploratory data analysis through visualization to reduce the computational complexity and enhance the training process, and hence performance of the learning algorithm.
3. Investigate on the different neural networks for Natural Language Processing (NLP) including MLP, CNN, RNN and LSTM and exploring their primary functionalities and benefits.
4. Implement a CNN-LSTM model for retrieval-based responses and an encoder-decoder LSTM model for generation of new responses. Adopt a hybrid approach through intent classification thereby implementing the better qualities of the models.
5. Evaluating the implemented language models by metrics such as the BLEU score and METEOR score in order to define a measure of success. Further assessing the final model based on user tests and a preconceived usability questionnaire riveted towards fluency, empathy and relevancy.


## Running the conversational agent
####Pre-requisite: Python 3.7 or newer (https://www.python.org/downloads/windows/)
The conversational agent can be run on a web browser using local host. 
The following steps may be used to do so.

1. Open terminal and change the directory to the folder containing app.py
```commandline
cd /path/to/file
```
2. Install required packages using pip and requirements.txt file
```commandline
pip install -r requirements.txt
```
3. Run app.py
```commandline
python app.py
```
4. Load the local host address into a web browser
```commandline
http://127.0.0.1:5000/ 
```

## Training the models
Training emotion classifier model
```commandline
python train_emotion_classifier.py
```

Training intent classifier model
```commandline
python train_intent_classifer.py
```
Training tag classifier model
```commandline
python train_tag_classifier.py
```

Training response generation model
```commandline
python seq2seq_model.py
```
## User Interface

## File Structure
```commandline
COMP3003_EmpathBot
├── README.md
├── requirements.txt
├── data
│   ├── counsel_chat.csv
│   ├── small_talk.json
│   ├── empathetic_dialogues_train.csv
│   ├── cc_vocab.pickle
│   ├── user_intent.csv
│   └── evaluation_data.csv
│
├── models
│   ├── classification
│   │   ├── emotion
│   │   ├── intent
│   │   └── tag
│   │
│   └── response_generation
│ 
├── glove.6B
│   └── glove.6B.100d.txt
│ 
├── static
│   ├── app.js
│   ├── chatBot.css
│   ├── ...
│   ├── ...
│   
├── templates
│   └── index.html
├── app.py
├── chat.py
├── data_preprocessing.py
├── database_handler.py
├── seq2seq_model.py
├── train_emotion_classifier.py
├── train_tag_classifier.py
├── train_intent_classifier.py
├── text_similarity.py
├── text_summarisation.py
├── evaluation_metric.py
├── fallback_data.aiml
├── std-startup.xml
└── data_visualisation.ipynb
```


## Dataset Used
Empathetic Dialogues: https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz

Counsel Chat: https://github.com/nbertagnolli/counsel-chat/blob/master/data/20200325_counsel_chat.csv

Small Talk: https://www.kaggle.com/datasets/elvinagammed/chatbots-intent-recognition-dataset

## Pre-trained Glove Embedding
https://nlp.stanford.edu/data/glove.6B.zip
# EmpathBot
