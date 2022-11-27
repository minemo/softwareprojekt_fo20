import pickle

import pandas as pd
import numpy as np
import torch
import nltk
import demoji
import os

from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from model import InductiveClusterer
from torch.utils.data import DataLoader
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


def get_emoji_meaning(text: str):
    # TODO: Use emoji semantics to get the meaning of an emoji
    # categorize emojis
    # 0: positive
    # 1: negative
    # 2: neutral
    # 3: female
    # 4: male

    emojs = list(demoji.findall(text))
    out = 2.0
    if emojs:
        vals = []
        for e in emojs:
            if '❤' or '‍🤝' in e:
                vals.append(0.0)
            elif '‍♂' in e:
                vals.append(4.0)
            elif '‍♀' in e:
                vals.append(3.0)
            elif '😂' in e:
                vals.append(2.0)
            elif '😭' in e:
                vals.append(1.0)
            else:
                vals.append(2.0)
        out = sum(vals) / len(vals)
    return out


def cross_validation_split(dataframe: pd.DataFrame, n_splits=5):
    """Split a dataframe into k folds for cross validation"""
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    folds = np.array_split(dataframe, n_splits)
    splits = []
    for i in range(n_splits):
        test = folds[i]
        train = pd.concat(folds[:i] + folds[i + 1:])
        splits.append((train, test, i))
    return splits


def main(debug=False, use_k_fold=True, save_model=False):
    """Main function"""

    if debug:
        print("Debug mode is enabled")
    else:
        print("Debug mode is disabled")

    # Load the data
    df = load_tsv("data.tsv")

    # Print some information about the data
    if debug:
        print('-' * 80)
        print(df.head())
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
        splits = cross_validation_split(df, n_splits=5)

        print('-' * 80)
        print('Training...')
        for train, test, i in splits:
            print(f'Fold {i + 1}')
            train_knn(best, debug, df, test, train)

    else:
        train = df.sample(frac=0.8)
        test = df.drop(train.index)
        train_knn(best, debug, df, test, train)

    # save the best model
    if save_model:
        with open('model.pkl', 'wb') as f:
            pickle.dump(best, f)
            f.close()


def train_knn(best, debug, df, test, train):
    X = np.array(
        [[get_emoji_meaning(text), count_mentions(text)] for text in train['c_text']])
    Y = np.array(train['hatespeech'])
    cluster = best[0]
    cluster.fit(X, Y)
    clf = best[1]
    if len(best) < 3:
        best.append(InductiveClusterer(cluster, clf))
    indl = best[2].fit(X, Y)
    # score is distance of prediction to actual value
    train_score = indl.score(X, Y)
    print(f'Accuracy: {train_score}')
    # test with the test set
    print('Testing...')
    X = np.array([[get_emoji_meaning(text), count_mentions(text)] for text in test['c_text']])
    Y = np.array(test['hatespeech'])
    test_score = indl.score(X, Y)
    print(f'Accuracy: {test_score}')
    # print some example classifications
    if debug:
        cluster = best[0]
        clf = best[1]
        indl = best[2].fit(X)
        print('Example classifications')
        predictions = []
        for text in df['c_text'].sample(100):
            predictions.append([indl.predict([[get_emoji_meaning(text), count_mentions(text)]])[0],
                                df[df["c_text"] == text]["hatespeech"].values[0]])
        print(f'Got {sum([1 for i in predictions if i[0] == i[1]])} correct out of {len(predictions)} total')
    print('-' * 80)


if __name__ == '__main__':
    # main(True)

    for i in [1390025325782917120, 1387006439160438785, 1390750616763441152, 1403975586838683652, 1395809007097503747,
              1389408533104545796]:
        print(extract_content(i))
