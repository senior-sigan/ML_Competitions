import numpy as np
import pandas as pd
from nltk import corpus
from sklearn.base import BaseEstimator, ClassifierMixin, clone, TransformerMixin
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
import categories


class Estimator(BaseEstimator, ClassifierMixin):
    """Estimator for this task
    Notes
    ----
    Predict subcategories, categories and final labels
    """

    def __init__(self, estimators, n_jobs=1):
        self.estimators = estimators
        self.n_jobs = n_jobs
        self.steps = ['low_category', 'category', 'GROUP_ID']
        self.selectors = [['NAME'], ['NAME', 'low_category'], ['NAME', 'low_category', 'category']]
        self.predictions_ = []
        self.mapper = categories.group_id()

    def fit(self, X, y=None):
        print("Fitting")
        # self.estimators_ = Parallel(n_jobs=self.n_jobs)(
        #     delayed(_parallel_fit_estimator)(clone(clf), X[self.selectors[i]], y, None)
        #     for i, clf in enumerate(self.estimators))

        self.estimators_ = []
        for i, clf in enumerate(self.estimators):
            print("\tFitting {}".format(self.selectors[i]))
            mapper = self.mapper[self.steps[i]]
            self.estimators_.append(clone(clf).fit(X[self.selectors[i]], y.apply(lambda row: mapper[row])))

        return self

    def predict(self, X):
        check_is_fitted(self, 'estimators_')
        print("Predicting:")
        return self._predict(X[['NAME']])  # just to save us on cross validation

    def _predict(self, X, i=0):
        print("\tPredicting step {}. Columns: {}".format(self.steps[i], X.columns))
        predictions = self.estimators_[i].predict(X)
        self.predictions_.append((self.steps[i], predictions))
        if i == len(self.estimators_) - 1:
            return predictions
        else:
            X_ = X.copy()
            X_[self.steps[i]] = predictions
            return self._predict(X_, i + 1)


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        return data[self.key]


class DummySelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformer = LabelBinarizer(sparse_output=True)

    def fit(self, X, y=None):
        self.transformer.fit(X)
        return self

    def transform(self, data):
        return self.transformer.transform(data)


def load_train_data():
    train = pd.read_csv('data/evo_train_clean.csv.gz', compression='gzip', index_col='id')
    categories = pd.read_csv('data/categories_parsed.csv.gz', compression='gzip', index_col='GROUP_ID')
    data = train.join(categories, on='GROUP_ID')
    mapper = {'Алкоголь': 'Прод', 'Прод': 'Прод', 'Непрод': 'Непрод', 'н/д': 'Непрод',
              'Позиция по свободной цене': 'Непрод'}
    data['low_category'] = data['category'].apply(lambda row: mapper[row])
    return data


def build_estimators():
    stopwords = corpus.stopwords.words('russian') + corpus.stopwords.words('english')
    low_category_clf = Pipeline([
        ('selector', ItemSelector(key='NAME')),
        ('count_vectorizer', CountVectorizer(
            binary=True,
            ngram_range=(1, 2),
            max_features=15000,
            stop_words=stopwords)),
        ('clf', VotingClassifier(estimators=[
            ('nb', MultinomialNB(alpha=0.2)),
            ('lr', LogisticRegression(class_weight='balanced', C=10, n_jobs=1))
        ], n_jobs=1, voting='soft'))
    ])
    sub_category_clf = Pipeline([
        ('features', FeatureUnion(transformer_list=[
            ('text', Pipeline([
                ('selector', ItemSelector(key='NAME')),
                ('count_vectorizer', CountVectorizer(
                    binary=True,
                    ngram_range=(1, 2),
                    max_features=15000,
                    stop_words=stopwords)),
            ])),
            ('low_category', Pipeline([
                ('selector', ItemSelector(key='low_category')),
                ('label', DummySelector())
            ]))
        ])),
        ('clf', VotingClassifier(estimators=[
            ('nb', MultinomialNB(alpha=0.2)),
            ('lr', LogisticRegression(class_weight='balanced', C=10, n_jobs=1))
        ], n_jobs=1, voting='soft'))
    ])
    category_clf = Pipeline([
        ('features', FeatureUnion(transformer_list=[
            ('text', Pipeline([
                ('selector', ItemSelector(key='NAME')),
                ('count_vectorizer', CountVectorizer(
                    binary=True,
                    ngram_range=(1, 2),
                    max_features=15000,
                    stop_words=stopwords)),
            ])),
            ('low_category', Pipeline([
                ('selector', ItemSelector(key='low_category')),
                ('label', DummySelector())
            ])),
            ('category', Pipeline([
                ('selector', ItemSelector(key='category')),
                ('label', DummySelector())
            ]))
        ])),
        ('clf', VotingClassifier(estimators=[
            ('nb', MultinomialNB(alpha=0.2)),
            ('lr', LogisticRegression(class_weight='balanced', C=10, n_jobs=1))
        ], n_jobs=1, voting='soft'))
    ])

    return Estimator([low_category_clf, sub_category_clf, category_clf])


def main():
    data = load_train_data()
    test = pd.read_csv('data/evo_test_clean.csv.gz', compression='gzip', index_col='id')
    X = data[['NAME', 'low_category', 'category']]
    y = data['GROUP_ID']
    clf = build_estimators()
    # scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='accuracy')
    # print("Accuracy: {}, std: {}".format(np.mean(scores), np.std(scores) * 2))
    clf.fit(X, y)
    pred = clf.predict(test)
    prediction = pd.DataFrame()
    prediction['id'] = test.index
    prediction['GROUP_ID'] = pred
    prediction.to_csv("submissions/submission-2.csv", index=False)
    # FAIL 0.916219


if __name__ == '__main__':
    main()
