import datetime
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from random import shuffle
from sklearn.model_selection import train_test_split


def main():
    travel = pd.read_csv('data/travel.csv.zip', index_col='id')
    prepare_data(travel)
    labels = prepare_labels(travel)
    model = train(labels)


def prepare_text(text):
    clean_text = re.sub("<.*?>", "", text).lower()
    splitter = re.compile("[^a-zA-Z0-9_\\+\\-/]")
    words = splitter.split(clean_text)
    stops = set(stopwords.words("english"))
    meaningful_words = [w.strip() for w in words if not w in stops]
    return " ".join(filter(None, meaningful_words))


def prepare_data(df):
    df['content'] = df['content'].apply(prepare_text)
    df['title'] = df['title'].apply(prepare_text)


def prepare_labels(df):
    sentences = []
    for i, row in df.iterrows():
        tags = tokenize_text(row['tags'])
        sentences.append(TaggedDocument(words=tokenize_text(row['content']), tags=tags))
    return sentences


def count_tags(df):
    tags = set()
    df['tags'].str.split(' ').apply(tags.update)
    return tags


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


def jaccard_similarity(labels, preds):
    lset = set(labels)
    pset = set(preds)
    return len(lset.intersection(pset)) / len(lset.union(pset))


def test(test_sents, model):
    results = []
    for test_sent in test_sents:
        pred_vec = model.infer_vector(test_sent.words)
        pred_tags = model.docvecs.most_similar([pred_vec], topn=5)
        results.append(jaccard_similarity(test_sent.tags, [p[0] for p in pred_tags]))
    return np.array(results)


def train(sentences):
    model = Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=4)
    model.build_vocab(sentences)
    train_sents, test_sents = train_test_split(sentences, test_size=0.2, random_state=42)
    alpha = 0.025
    min_alpha = 0.001
    num_epochs = 200
    alpha_delta = (alpha - min_alpha) / num_epochs

    for epoch in range(num_epochs):
        start_time = datetime.datetime.now()
        shuffle(train_sents)
        model.alpha = alpha
        model.min_alpha = alpha
        model.train(train_sents)
        alpha -= alpha_delta
        end_time = datetime.datetime.now()
        accuracy = test(test_sents, model).mean()
        print("Complete epoch {}: {}; Accuracy: {}".format(epoch, end_time - start_time, accuracy))

    return model

if __name__ == '__main__':
    main()
