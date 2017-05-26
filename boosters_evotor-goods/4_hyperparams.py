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

stopwords = corpus.stopwords.words('russian') + corpus.stopwords.words('english')


def load_data():
    train = pd.read_csv('data/evo_train_clean.csv.gz', compression='gzip', index_col='id')
    categories = pd.read_csv('data/categories_parsed.csv.gz', compression='gzip', index_col='GROUP_ID')
    data = train.join(categories, on='GROUP_ID')
    mapper = {'Алкоголь': 'Прод', 'Прод': 'Прод', 'Непрод': 'Непрод', 'н/д': 'Непрод',
              'Позиция по свободной цене': 'Непрод'}
    data['supercategory'] = data['category'].apply(lambda row: mapper[row])
    data.head()
    return data


def prepare_data_baseline(data):
    X = data['NAME']
    y = data['supercategory']

    return train_test_split(X, y, test_size=0.3, random_state=42)


def report(clf, parameters, scoring='accuracy'):
    data = load_data()
    X_train, X_test, y_train, y_test = prepare_data_baseline(data)
    search = GridSearchCV(clf, parameters, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring=scoring, n_jobs=2,
                          verbose=2)
    search.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(search.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = search.cv_results_['mean_test_score']
    stds = search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, search.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


def train_report(clf):
    data = load_data()
    X_train, X_test, y_train, y_test = prepare_data_baseline(data)

    # Cross validation
    # X = data['NAME']
    # y = data['supercategory']
    # scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='neg_log_loss')
    # print("LogLoss: {}, std: {}".format(np.mean(scores), np.std(scores) * 2))

    # Train and test on holdout with report
    clf.fit(X_train, y_train)
    y_true, y_pred = y_test, clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    print("Accuracy: {}".format(accuracy_score(y_true, y_pred)))
    print("LogLoss: {}".format(log_loss(y_true, y_pred_proba)))
    print(classification_report(y_true, y_pred))

    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cnf_matrix, y_true.unique())


def prepare_logistic():
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(
            binary=False,
            ngram_range=(1, 2),
            max_features=15000,
            stop_words=stopwords)),
        ('clf', LogisticRegression(random_state=42))
    ])
    parameters = {
        'clf__C': [0.5, 1, 5, 8, 10, 12, 15, 20],
        'clf__class_weight': ['balanced'],
        'clf__penalty': ['l2', 'l1']
    }
    report(pipeline, parameters, 'neg_log_loss')


def prepare_nb():
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(
            binary=True,
            ngram_range=(1, 2),
            max_features=15000,
            stop_words=stopwords)),
        ('clf', MultinomialNB())
    ])
    parameters = {
        'clf__alpha': [0, 0.2, 0.3, 0.5, 0.5, 0.7, 1, 1.2]
    }
    report(pipeline, parameters, 'accuracy')


def prepare_rf():
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(
            binary=True,
            ngram_range=(1, 2),
            max_features=15000,
            stop_words=stopwords)),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=2))
    ])
    parameters = {
        'clf__n_estimators': [10],
        'clf__max_depth': [None],
        'clf__max_features': [None, 'auto', 'log2'],
        'clf__class_weight': ['balanced']
    }
    report(pipeline, parameters, 'neg_log_loss')


def ada():
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(
            binary=True,
            ngram_range=(1, 2),
            max_features=15000,
            stop_words=stopwords)),
        ('clf', AdaBoostClassifier(

        ))
    ])
    train_report(pipeline)


def ensemble():
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
    train_report(pipeline)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    print("MAIN")
    # prepare_logistic()
    # prepare_nb()
    # prepare_rf()
    # ensemble()
    # ada()
