'''
Summarise lengthy answers in counsel chat dataset
'''
import pandas as pd
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from nltk import tokenize

def get_sentences(answer):
    answer = tokenize.sent_tokenize(answer)
    sentences = []
    for sentence in answer:
        # sentence = sentence.replace(".", "")
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    return sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(sentences, top_n=5):
    stop_words = stopwords.words('english')
    text_summary = []

    # Generate similarity matrix
    similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # Rank sentences
    similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(similarity_graph)

    # Fetch top sentences
    # for i, s in enumerate(sentences):
    #    ranked_sentence = (scores[i], s)

    # sorted_ranked_sentence = sorted(ranked_sentence, reverse=True)
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    # print("Indexes of top ranked_sentence order are ", ranked_sentence)
    # print("\n")

    for i in range(top_n):
        text_summary.append(" ".join(ranked_sentence[i][1]))

    return text_summary

# Loading counsel dataset
counsel_chat_df = pd.read_csv('data/counsel_chat.csv', encoding='utf-8', low_memory=False)
# counsel_chat_df['responseLength'] = counsel_chat_df['answerText'].apply(lambda x: len(str(x).split(' ')))
answerSummary = []

for row in counsel_chat_df.iterrows():
    answer = row[1]['answerText']
    sentences = get_sentences(answer) # get list of sentences
    if len(sentences) > 1:
        # print("Text to be summarised: ", sentences )
        # print("\n")
        summary = generate_summary(sentences, 2)
        # print('Summary: ', summary)
        answerSummary.append(". ".join(summary))
    else:
        answerSummary.append(answer)

counsel_chat_df['answerSummary'] = answerSummary

# adding column answer summary to counsel chat dataset
counsel_chat_df.to_csv('data/counsel_chat.csv')
print('Answer summary added to counsel chat dataset')














