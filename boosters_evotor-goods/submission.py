import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk import corpus
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def load_train_data():
    train = pd.read_csv('data/evo_train_clean.csv.gz', compression='gzip', index_col='id')
    categories = pd.read_csv('data/categories_parsed.csv.gz', compression='gzip', index_col='GROUP_ID')
    data = train.join(categories, on='GROUP_ID')
    mapper = {'Алкоголь': 'Прод', 'Прод': 'Прод', 'Непрод': 'Непрод', 'н/д': 'Непрод',
              'Позиция по свободной цене': 'Непрод'}
    data['supercategory'] = data['category'].apply(lambda row: mapper[row])
    data.head()
    return data


def ensemble():
    stopwords = corpus.stopwords.words('russian') + corpus.stopwords.words('english')
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(
            binary=True,
            ngram_range=(1, 2),
            max_features=15000,
            stop_words=stopwords)),
        ('clf', VotingClassifier(estimators=[
            ('nb', BaggingClassifier(MultinomialNB(alpha=0.2))),
            ('lr', BaggingClassifier(LogisticRegression(class_weight='balanced', C=10, n_jobs=2))),
            # ('rf', RandomForestClassifier(n_estimators=200, max_features='log2', class_weight='balanced', n_jobs=2))
        ], n_jobs=2, voting='soft', weights=[1, 1]))
    ])
    return pipeline