import pickle

import nltk
import pandas as pd
import numpy as np
import demoji
import os

import torch
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from torch import optim, nn

from model import InductiveClusterer, LNN, LnnDataset
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
    """Count the number of words in a tweet using the nltk tweet tokenizer"""
    return len(nltk.tokenize.TweetTokenizer().tokenize(text))


def get_link_similarity(tid, text):
    """Calculates the similarity between the text and the most relevant words of the website"""
    relevant_words = []
    try:
        relevant_words = extract_content(tid)
        if relevant_words is [] or relevant_words is None:
            return 0
        else:
            return nltk.jaccard_distance(set(nltk.tokenize.TweetTokenizer().tokenize(text)), relevant_words[0].keys())
    except Exception as e:
        print(f'Error while extracting content for {tid}: {e}')
        return 0


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
        count_emojitexts = dataset[dataset["c_text"].apply(lambda x: len(demoji.findall(x))) > 0].shape[0]
        # count tweets with hashtags
        nhashtags = dataset["features"].apply(lambda x: x[2]).sum()
        # count tweets with mentions
        nmentions = dataset["features"].apply(lambda x: x[1]).sum()
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
    df["features"] = df["c_text"].apply(lambda x: [get_emoji_meaning(x, emojidata),
                                                   count_mentions(x),
                                                   count_hashtags(x),
                                                   count_words(x)])

    # add likecount and retweetcount to the features
    df["features"] = df.apply(lambda x: x["features"] + [x["like_count"], x["retweet_count"]], axis=1)

    # remove entries where the target is empty
    df = df[df["target"].notna()]

    # Print some information about the data
    if debug:
        save_dataset_features(df, "dataset_features.tex")
        print('-' * 80)
        print(df.head(3))
        print(df.describe())
        print(df.info())

    insize = 4  # number of features
    hidden = 16  # number of hidden units
    outsize = 4  # number of output units (target classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lnn_model = (LNN(insize, hidden, outsize).to(device), device)
    hatespeechmodel = [AgglomerativeClustering(), RandomForestClassifier(n_jobs=-1), lnn_model]

    if os.path.exists("model.pkl"):
        print("Found model.pkl, loading it...")
        with open("model.pkl", "rb") as f:
            hatespeechmodel = pickle.load(f)
        print(f'Loaded Model is: {hatespeechmodel}')
        print(f'Number of clusters: {hatespeechmodel[0].n_clusters_}')
        print(f'Current loss is: {hatespeechmodel[3].best_score_ if len(hatespeechmodel) > 3 else "N/A"}')
    else:
        print("No existing model, training from scratch...")

    if use_k_fold:
        # shuffle the data and split it into 5 folds
        df = df.sample(frac=1).reset_index(drop=True)
        splits = cross_validation_split(df, n_splits=5)

        print('-' * 80)
        print('Training...')
        for train, test, i in splits:
            # train the clustering model
            print(f'Fold {i + 1}')
            train_knn(hatespeechmodel, debug, df, test, train)

            # train the linear neural network
            # append the corresponding hatespeech label to the feature vector
            train["features"] = train.apply(lambda x: x["features"] + [x["hatespeech"]], axis=1)
            test["features"] = test.apply(lambda x: x["features"] + [x["hatespeech"]], axis=1)
            print(train.shape, test.shape)
            # train_linearNN(best, debug, df, LnnDataset(test), LnnDataset(train))

    else:
        train = df.sample(frac=0.8)
        test = df.drop(train.index)
        # train_knn(best, debug, df, test, train)

        train["features"] = train.apply(lambda x: x["features"] + [x["hatespeech"]], axis=1)
        test["features"] = test.apply(lambda x: x["features"] + [x["hatespeech"]], axis=1)
        # print(train, test)
        train_linearNN(hatespeechmodel, debug, df, LnnDataset(test), LnnDataset(train))

    # save the best model
    if save_model:
        with open('model.pkl', 'wb') as f:
            pickle.dump(hatespeechmodel, f)
            f.close()


def train_linearNN(best, debug, df, test: LnnDataset, train: LnnDataset, epochs=100):
    model, device = best[2]

    # convert the training data to a DataLoader
    train_dl = DataLoader(train, batch_size=1, shuffle=True)
    test_dl = DataLoader(test, batch_size=1, shuffle=True)

    # train the model
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            # convert yb to one-hot
            yb = torch.nn.functional.one_hot(yb, num_classes=4).float()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                # convert yb to one-hot
                yb = torch.nn.functional.one_hot(yb, num_classes=4).float()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                test_loss.append(loss.item())

        if debug:
            print(f'Epoch {epoch + 1}: {loss.item()}')


def train_knn(best, debug, df, test, train):
    # define the data
    x = np.array(list(train["features"]))
    hy = np.array(train['hatespeech'])

    # train the clustering model
    cluster = best[0]
    cluster.fit(x, hy)
    clf = best[1]
    if len(best) < 4:
        best.append(InductiveClusterer(cluster, clf))
    indl = best[3].fit(x, hy)

    hate_train_score = indl.score(x, hy)
    print(f'Hatespeech accuracy: {hate_train_score}')
    # test with the test set
    print('Testing...')
    x = np.array(list(test["features"]))
    hy = np.array(test['hatespeech'])
    ty = np.array([["person", "group", "public"].index(i) for i in test['target']])
    hate_test_score = indl.score(x, hy)
    print(f'Hatespeech accuracy: {hate_test_score}')
    # print some example classifications
    if debug:
        indl = best[3].fit(x, hy)
        print('-' * 20 + 'Example Classifications' + '-' * 20)
        predictions = []
        for i in range(100):
            data = df.sample(1)
            x = np.array(list(data["features"])).reshape(1, -1)
            hy = df["hatespeech"].iloc[i]
            prediction = indl.predict(x)
            predictions.append((hy, prediction))
        print(f'Got {len([x for x in predictions if x[0] == x[1]])} correct predictions out of 100')

    print('-' * 80)


if __name__ == '__main__':
    main(True, True, True)


def output(dataframe: pd.DataFrame):
    headerList = ['c_id', 'hatespeech', 'target']
    targets = ["person", "group", "public"]
    dataframe["target"] = [targets[ele] if ele <= len(targets) else "none" for ele in dataframe["target"]]
    return dataframe.to_csv('Target_Gruppe6_1a.csv',header=headerList, index=False)
