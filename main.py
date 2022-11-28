import pickle

import nltk
import pandas as pd
import numpy as np
import demoji
import os

from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from model import InductiveClusterer
from linkscraping import extract_content


def load_tsv(path: str):
    """Load a tsv file into a pandas dataframe"""
    return pd.read_csv(path, sep="\t")


def count_mentions(text: str):
    """Count the number of mentions in a tweet"""
    return text.count("@")


def count_hashtags(text: str):
    """Count the number of hashtags in a tweet"""
    return text.count("#")


def count_words(text: str):
    """Count the number of words in a tweet using the nltk library"""
    return len(nltk.word_tokenize(text))


def get_emoji_meaning(text: str, emojidata: pd.DataFrame):
    emojs = list(demoji.findall(text))
    # calculate the mean sentiment of the emojis
    sentiment = 0
    for emo in emojs:
        if emo in emojidata["Emoji"].values:
            positive = int(emojidata[emojidata['Emoji'] == emo]["Positive"])
            negative = int(emojidata[emojidata['Emoji'] == emo]["Negative"])
            neutral = int(emojidata[emojidata['Emoji'] == emo]["Neutral"])
            # calculate average sentiment (positive is 1, negative is -1, neutral is 0)
            sentiment += (positive - negative) / (positive + negative + neutral)
    return (sentiment / len(emojs)) if len(emojs) > 0 else 0.0


def save_dataset_features(dataset: pd.DataFrame, path: str):
    """Save the features of a dataset to a file"""
    with open(path, "w") as f:
        # save as latex table

        # count tweets with emojis
        count_emojitexts = dataset[dataset["emoji_sentiment"] != 0].shape[0]
        # count tweets with hashtags
        nhashtags = dataset[dataset["hashtags"] != 0].shape[0]
        # count tweets with mentions
        nmentions = dataset[dataset["mentions"] != 0].shape[0]
        # count tweets with urls
        count_urls = dataset[dataset["c_text"].str.contains("http")].shape[0]

        outdf = pd.DataFrame()
        outdf["Feature"] = ["Tweets die Emojis enthalten",
                            "Tweets die Hashtags enthalten",
                            "Tweets die andere Nutzer erwähnen",
                            "Tweets die URLs enthalten"]
        outdf["Anzahl"] = [count_emojitexts, nhashtags, nmentions, count_urls]
        outdf.to_latex(f, index=False)


def cross_validation_split(dataframe: pd.DataFrame, n_splits=5):
    """Split a dataframe into k folds for cross validation"""
    folds = list()
    dataframe_split = np.array_split(dataframe, n_splits)
    for i in range(n_splits):
        test = dataframe_split[i]
        train = pd.concat([x for x in dataframe_split if x is not test])
        folds.append((train, test, i))
    return folds


def main(debug=False, use_k_fold=True, save_model=False):
    """Main function"""

    if debug:
        print("Debug mode is enabled")
    else:
        print("Debug mode is disabled")

    # Load the data
    df = load_tsv("data.tsv")
    emojidata = pd.read_csv("emoji_sentiment.csv")

    # append features to the dataframe
    df["emoji_sentiment"] = df["c_text"].apply(lambda x: get_emoji_meaning(x, emojidata))
    df["mentions"] = df["c_text"].apply(lambda x: count_mentions(x))
    df["hashtags"] = df["c_text"].apply(lambda x: count_hashtags(x))

    # remove entries where the target is empty
    df = df[df["target"].notna()]

    # Print some information about the data
    if debug:
        save_dataset_features(df, "dataset_features.tex")
        print('-' * 80)
        print(df.head(3))
        print(df.describe())
        print(df.info())

    best = [AgglomerativeClustering(), RandomForestClassifier(n_jobs=-1)]
    if os.path.exists("model.pkl"):
        print("Found model.pkl, loading it...")
        with open("model.pkl", "rb") as f:
            best = pickle.load(f)
        print(f'Loaded Model is: {best}')
        print(f'Number of clusters: {best[0].n_clusters_}')
        print(f'Current loss is: {best[2].best_score_}')
    else:
        print("No existing model, training from scratch...")

    if use_k_fold:
        # shuffle the data and split it into 5 folds
        df = df.sample(frac=1).reset_index(drop=True)
        splits = cross_validation_split(df, n_splits=5)

        print('-' * 80)
        print('Training...')
        for train, test, i in splits:
            print(f'Fold {i + 1}')
            train_knn(best, debug, df, test, train)

    else:
        train = df.sample(frac=0.8)
        test = df.drop(train)
        train_knn(best, debug, df, test, train)

    # save the best model
    if save_model:
        with open('model.pkl', 'wb') as f:
            pickle.dump(best, f)
            f.close()


def train_knn(best, debug, df, test, train):
    x = np.array(
        [np.array([x, y, z]) for x, y, z in zip(train["emoji_sentiment"], train["mentions"], train["hashtags"])])
    y = np.array(train['hatespeech'])
    cluster = best[0]
    cluster.fit(x, y)
    clf = best[1]
    if len(best) < 3:
        best.append(InductiveClusterer(cluster, clf))
    indl = best[2].fit(x, y)
    # score is distance of prediction to actual value
    train_score = indl.score(x, y)
    print(f'Accuracy: {train_score}')
    # test with the test set
    print('Testing...')
    x = np.array(
        [np.array([x, y, z]) for x, y, z in zip(test["emoji_sentiment"], test["mentions"], test["hashtags"])])
    y = np.array(test['hatespeech'])
    test_score = indl.score(x, y)
    print(f'Accuracy: {test_score}')
    # print some example classifications
    if debug:
        indl = best[2].fit(x, y)
        print('-' * 20 + 'Example Classifications' + '-' * 20)
        predictions = []
        for i in range(100):
            data = df.iloc[i]
            x = np.array(
                [np.array([data["emoji_sentiment"], data["mentions"], data["hashtags"]])])
            y = df["hatespeech"].iloc[i]
            prediction = indl.predict(x)
            predictions.append((y, prediction))
        print(f'Got {len([x for x in predictions if x[0] == x[1]])} correct predictions out of 100')

    print('-' * 80)


if __name__ == '__main__':
    main(True, True, True)
