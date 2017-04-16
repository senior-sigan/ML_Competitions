from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class Classifier:
    def __init__(self, tags):
        self.tags = tags
        self.clf_list = []

    def fit(self, x, y):
        for tag in self.tags:
            start_time = datetime.now()
            clf = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier())])
            clf.fit(x, y[tag])
            print("{}: {}".format(tag, datetime.now() - start_time))
            self.clf_list.append((tag, clf))

    def predict(self, x):
        pred_dic = {}
        for tag, clf in self.clf_list:
            pred_dic[tag] = clf.predict(x)
        return pred_dic


def join_pred(tagged_pred):
    df = pd.DataFrame(tagged_pred)
    res = pd.DataFrame(index=df.index)
    tags = tagged_pred.keys()
    for i, row in df.iterrows():
        tags = [tag for tag in tags if row[tag] == 1]
        res.ix[i, 'tags'] = ' '.join(tags)

    return res


def build(data, tags):
    start_time = datetime.now()
    clf = Classifier(tags)
    clf.fit(data['content'], data)
    print("Time to fit: {}".format(datetime.now() - start_time))
    return clf


def match(list_1, list_2):
    assert len(list_1) == len(list_2)

    matches = 0
    for i in range(len(list_1)):
        if list_1[i] == list_2[i]:
            matches += 1

    return matches / len(list_1)


def accuracy(clf, x_test, y_test):
    pred = clf.predict(x_test)
    scores = 0
    for tag in pred:
        score = match(y_test[tag].values, pred[tag])
        scores += score
        # print("Tag: {}. Score: {}".format(tag, score))
    return scores / len(pred)


def cross_validation(data, tags, n=5):
    total_score = 0

    for i in range(0, n):
        print("===={}=====".format(i))
        train_data, test_data = train_test_split(data, test_size=0.2)
        clf = build(train_data, tags)
        score = accuracy(clf, test_data['content'], test_data)
        total_score += score
        print("{}) Score: {}".format(i, score))
    print("======================")
    print("Total score: {}".format(total_score / n))


def build_all(files):
    start_time = datetime.now()
    clf_list = []
    for file in files:
        tags = pd.read_csv("prepared_data/tags_{}.csv.zip".format(file), index_col='id', compression='gzip')
        data = pd.read_csv("prepared_data/{}.csv.zip".format(file), index_col='id', compression='gzip')
        print("==========Data loaded for {}===========".format(file))
        n = tags.describe().ix['75%', 'count']
        tags = tags[tags['count'] > n].index

        # cross_validation(data, ['visas', 'uk', 'usa'])
        clf = build(data, tags)
        clf_list.append((file, clf))

    print("=============Training ends in: {}==============".format(datetime.now() - start_time))
    return clf_list


def main():
    test_data = pd.read_csv("prepared_data/test.csv.zip", index_col='id', compression='gzip')
    files = ['biology', 'cooking', 'crypto', 'robotics', 'travel']
    clf_list = build_all(files)
    predictions = pd.DataFrame(index=test_data.index)
    for file, clf in clf_list:
        started_time = datetime.now()
        print("Predicting {}".format(file))
        pred = clf.predict(test_data['content'].head(100))
        print(pred)
        print("Joining {}".format(file))
        predictions[file] = join_pred(pred)
        print("Predicted {}. Time: {}".format(file, datetime.now() - started_time))

    print(predictions.head())
    print("Will save")
    predictions.to_csv("prepared_data/predictions.csv")


if __name__ == '__main__':
    main()
