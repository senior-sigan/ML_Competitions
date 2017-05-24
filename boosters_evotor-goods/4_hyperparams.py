import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier


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
    search = GridSearchCV(clf, parameters, cv=StratifiedKFold(n_splits=5), scoring=scoring, n_jobs=3)
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


def prepare_logistic():
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(
            binary=False,
            ngram_range=(1, 2),
            max_features=15000,
            stop_words=stopwords.words('russian') + stopwords.words('english'))),
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
            stop_words=stopwords.words('russian') + stopwords.words('english'))),
        ('clf', MultinomialNB())
    ])
    parameters = {
        'clf__alpha': [0, 0.2, 0.3, 0.5, 0.5, 0.7, 1, 1.2]
    }
    report(pipeline, parameters, 'accuracy')


def prepare_knn():
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(
            binary=True,
            ngram_range=(1, 2),
            max_features=15000,
            stop_words=stopwords.words('russian') + stopwords.words('english'))),
        ('clf', KNeighborsClassifier())
    ])
    parameters = {
        'clf__n_neighbors': [1, 5, 7, 10, 15, 20],
        'clf__metric': ['manhattan', 'minkowski']
    }
    report(pipeline, parameters, 'accuracy')


if __name__ == '__main__':
    prepare_logistic()
    # prepare_nb()
    # prepare_knn()
