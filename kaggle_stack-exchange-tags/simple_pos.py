import itertools
import multiprocessing as mp
import re
import string
from datetime import datetime

import nltk
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from bs4 import BeautifulSoup


def main(data):
    punctuation = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    print("===Start predicting in {}===".format(mp.current_process().name))
    res = predict(data, chunker, punctuation, stop_words)
    print(res.shape)
    return res


def split(a, n):
    k, m = len(a) // n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def score_keyphrases_by_textrank(candidates, text, n_keywords=0.05):
    from itertools import takewhile, tee
    import networkx, nltk
    
    # tokenize for all words, and extract *candidate* words
    words = [word.lower()
             for sent in nltk.sent_tokenize(text)
             for word in nltk.word_tokenize(sent)]
    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)
    
    return [tag for tag, score in sorted(keyphrases.items(), key=lambda x: x[1], reverse=True)]


def select_best_tags(tags, text):
    # return list(tags)[0:5]
    return score_keyphrases_by_textrank(tags, text, 0.08)

def predict(test_data, chunker, punctuation, stop_words):
    res = pd.DataFrame(index=test_data.index)
    count = 0
    start_time = datetime.now()
    for i, row in test_data.iterrows():
        count += 1
        if count % 1000 == 0:
            print("{}) {} {}".format(mp.current_process().name, count / 1000, datetime.now() - start_time))
            start_time = datetime.now()
        body = "{} {}".format(test_data.ix[i, 'title'], test_data.ix[i, 'content'])
        #tags = extract_candidate_chunks(text=body, chunker=chunker, punctuation=punctuation, stop_words=stop_words)
        tags = extract_candidate_words(body)
        res.ix[i, 'tags'] = ' '.join(select_best_tags(tags, body))

    return res


def remove_html(text):
    # re.sub("[0-9]+", "#NUMBER#", text)  # test says it makes prediction worse
    return re.sub("[0-9]+", "", re.sub(r'\s+', ' ', BeautifulSoup(text, "html.parser").get_text()).lower().strip()).replace('"', '').replace('%', '').replace(';','.')


def prepare_test_data(df, func):
    res = pd.DataFrame(index=df.index)
    res['title'] = df['title'].apply(func)
    res['content'] = df['content'].apply(func)
    return res


def group_func(t):
    (word, pos, chunk) = t
    return chunk != 'O'


def extract_candidate_chunks(text, chunker, stop_words, punctuation):
    # tokenize, POS-tag, and chunk using regular expressions
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = [nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                  for tagged_sent in tagged_sents]
    all_chunks = list(itertools.chain.from_iterable(all_chunks))
    # join constituent chunk words into a single chunked phrase
    candidates = ['-'.join(word for word, pos, chunk in group).lower()
                  for key, group in itertools.groupby(all_chunks, group_func) if key]

    return set([cand for cand in candidates
                if cand not in stop_words and not all(char in punctuation for char in cand) and 2 < len(cand) < 15])


def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word) and 2 < len(tag) < 10]

    return candidates


def predict_train(file='data/test.csv.zip'):
    n_threads = 2
    pool = mp.Pool(processes=n_threads)
    test = pd.read_csv(file, index_col='id')
    test = prepare_test_data(test, remove_html)
    splits = [data for data in split(test, n_threads)]
    res = pool.map(main, splits)
    pool.close()
    pool.join()
    pd.concat(res).to_csv('submissions/simple_pos.csv', index_label='id')


def tags_f1_score(y_true, y_pred):
    all_data = list(set(y_true + y_pred))
    yt_dict = {}
    yp_dict = {}
    yt = [None] * len(all_data)
    yp = [None] * len(all_data)
    for tag in y_true:
        yt_dict[tag] = 1
    for tag in y_pred:
        yp_dict[tag] = 1
    for i in range(0, len(all_data)):
        tag = all_data[i]
        if yt_dict.get(tag, 0) == 1:
            yt[i] = 1
        else:
            yt[i] = 0
        if yp_dict.get(tag, 0) == 1:
            yp[i] = 1
        else:
            yp[i] = 0
    return f1_score(yt, yp)


def validate_test():
    data = pd.read_csv('data/robotics.csv.zip', index_col='id')
    test_data = data[0:1000]
    print(test_data.shape)
    pred = main(test_data)

    print("====Scoring====")
    scores = []
    for i, row in pred.iterrows():
        y_pred = pred.ix[i, 'tags'].split(' ')
        y_true = test_data.ix[i, 'tags'].split(' ')
        scores.append(tags_f1_score(y_pred, y_true))
    scores = np.array(scores)
    print(scores.mean())


def super_prepare():
    test = pd.read_csv('data/test.csv.zip', index_col='id')
    test = prepare_test_data(test, remove_html)
    test.to_csv('data/test_clear.csv', index_label='id')


if __name__ == '__main__':
    #validate_test()
    predict_train()
    #super_prepare()
