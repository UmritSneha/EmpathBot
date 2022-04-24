'''
Implementing cosine similarity
'''

# importing required libraries
import nltk
from nltk.cluster.util import cosine_distance

def cosine_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    tokenizer = nltk.RegexpTokenizer(r"\w+")

    text1 = tokenizer.tokenize(sent1)

    text2 = tokenizer.tokenize(sent2)

    text1 = [w.lower() for w in text1]
    text2 = [w.lower() for w in text2]

    vocab = list(set(text1 + text2))

    vector1 = [0] * len(vocab)
    vector2 = [0] * len(vocab)

    # build the vector for the first sentence
    for w in text1:
        if w in stopwords:
            continue
        vector1[vocab.index(w)] += 1

    # build the vector for the second sentence
    for w in text2:
        if w in stopwords:
            continue
        vector2[vocab.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

'''def cosine_similarity_archived(text1, text2, ngram):
    texts = [text1, text2]
    # print(texts)
    tokens = text_tokenizer(texts)
    # print(tokens)
    docs = stopword_removal(tokens)
    docs = document_stemmer(docs)
    # print(docs)

    # Generating a bigram document

    if ngram == 2:
        bigram_docs = []
        for doc in docs:
            bigram_docs.extend(list(ngrams(doc, 2)))
        docs = bigram_docs

    # Creating vocabulary
    vocab = []
    for doc in docs:
        for item in doc:
            if item not in vocab:
                vocab.append(item)

    # Creating a bag-of-word model
    bow = []
    for doc in docs:
         vector = np.zeros(len(vocab))
         for item in doc:
             index = vocab.index(item)
             vector[index] += 1
         bow.append(vector)

    # print(bow)
    query = bow[0]
    bow = dict(d1=bow[1])
    for d in bow.keys():
        sim = 1 - spatial.distance.cosine(query, bow[d])
    return sim'''

