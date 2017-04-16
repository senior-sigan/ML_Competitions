import pandas as pd


def prepare(file):
    df = pd.read_csv('data/{}.csv.zip'.format(file), index_col='id')
    print("Shape of {}: {}".format(file, df.shape))
    tag_freq = tags_frequencies(df)
    tags = pd.DataFrame(list(tag_freq.values()), columns=['count'], index=tag_freq.keys())
    tags = tags.sort_values(['count'], ascending=False)
    print(tags.head(10))
    print(tags.describe())
    return tags


def tags_frequencies(df):
    freq = {}
    for i, row in df.iterrows():
        tags = row['tags'].split(' ')
        for tag in tags:
            freq[tag] = freq.get(tag, 0) + 1

    return freq


def main():
    files = ['biology', 'cooking', 'crypto', 'robotics', 'travel']
    for file in files:
        print("===={}====".format(file))
        prepare(file).to_csv("prepared_data/tags_{}.csv.zip".format(file), index_label='id', compression='gzip')

if __name__ == '__main__':
    main()
