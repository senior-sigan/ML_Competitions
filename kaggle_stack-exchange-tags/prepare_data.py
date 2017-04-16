import pandas as pd
import re
from nltk.corpus import stopwords
from datetime import datetime


def prepare_text(text):
    clean_text = re.sub("<.*?>", "", text).lower()
    splitter = re.compile("[^a-zA-Z0-9_\\+\\-/]")
    words = splitter.split(clean_text)
    stops = set(stopwords.words("english"))
    meaningful_words = [w.strip() for w in words if not w in stops]
    return " ".join(filter(None, meaningful_words))


def prepare_data(df):
    res = pd.DataFrame(index=df.index)
    res['tags'] = df['tags']
    res['title'] = df['title'].apply(prepare_text)
    res['content'] = df['content'].apply(prepare_text)
    return res


def tags_bag(df, tags):
    print("Data shape: {}. Tags shape: {}".format(df.shape, tags.shape))
    start_time = datetime.now()
    bag = pd.DataFrame(index=df.index)
    bag['title'] = df['title']
    bag['content'] = df['content']
    counter = 0
    start_step_time = datetime.now()
    tag_indexes = tags.index
    total_count = df.shape[0]
    barrier = total_count // 10
    for i, row in df.iterrows():
        counter += 1
        for tag in row['tags'].split(' '):
            if tag in tag_indexes:
                bag.ix[i, tag] = 1
        if counter % barrier == 0:
            print("Step {}: {}".format(counter / total_count, datetime.now() - start_step_time))
            start_step_time = datetime.now()
    print("Time: {}".format(datetime.now() - start_time))
    return bag.fillna(0)


def prepare(file):
    df = pd.read_csv('data/{}.csv.zip'.format(file), index_col='id')
    print(df.shape)
    tags_freq = pd.read_csv('prepared_data/tags_{}.csv.zip'.format(file), index_col='id', compression='gzip')
    df = prepare_data(df)
    n = tags_freq.describe().ix['75%', 'count']
    print("Tag frequency limit: {}".format(n))
    df = tags_bag(df, tags_freq[tags_freq['count'] > n])
    return df


def prepare_test(file):
    df = pd.read_csv('data/{}.csv.zip'.format(file), index_col='id')
    res = pd.DataFrame(index=df.index)
    res['title'] = df['title'].apply(prepare_text)
    res['content'] = df['content'].apply(prepare_text)
    return res


def main():
    files = ['biology', 'cooking', 'crypto', 'robotics', 'travel']
    for file in files:
        print("===={}====".format(file))
        prepare(file).to_csv('prepared_data/{}.csv.zip'.format(file), index_label='id', compression='gzip')

    print("====test=====")
    prepare_test('test').to_csv('prepared_data/test.csv.zip', index_label='id', compression='gzip')


if __name__ == '__main__':
    main()
