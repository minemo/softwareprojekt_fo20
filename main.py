import pandas as pd
import numpy as np
import torch
import nltk
import demoji
import emoji
import os

from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from model import InductiveClusterer


# from model import TwitterHatespeechModel


def load_tsv(path: str):
    """Load a tsv file into a pandas dataframe"""
    return pd.read_csv(path, sep="\t")


class TweetTokenizer:
    """A tokenizer for tweets with the ability to decode emojis"""

    # TODO: use get_Link_info when its implemented to add the link information to the tweet

    def __init__(self):
        self.tokenizer = nltk.tokenize.TweetTokenizer()
        self.vocab = []

    def generate_vocab(self, texts: list[str]):
        """Generate a vocabulary from a list of tweets"""
        tokens = []
        for text in texts:
            tokens += self.encode(text)
        self.vocab = list(set(tokens))
        self.vocab.sort()
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + self.vocab

    def get_Link_info(self, text: str):
        """Get the information about the contents of a link in a tweet"""
        # TODO: Implement this function
        raise NotImplementedError

    def encode(self, text: str, max_len=50):
        """Encode a tweet into a list of tokens"""
        text = demoji.replace_with_desc(text, sep=" ")
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:max_len]
        return tokens

    def encode_plus(self, text: str, max_len=50, add_special_tokens=True, padding="max_length",
                    return_attention_mask=True,
                    return_tensors="pt"):
        """Encode a Tweet and return a dictionary of tensors"""
        tokens = self.encode(text, max_len)
        if add_special_tokens:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
        if padding == "max_length":
            tokens = tokens + ["[PAD]"] * (max_len - len(tokens))
        token_ids = [self.vocab.index(token) for token in tokens]
        token_ids = torch.tensor(token_ids)
        if return_attention_mask:
            attention_mask = token_ids != self.vocab.index("[PAD]")
            attention_mask = attention_mask.to(torch.long)
        if return_tensors == "pt":
            token_ids = token_ids.to(torch.long)
        output = {"input_ids": token_ids}
        if return_attention_mask:
            output["attention_mask"] = attention_mask
        return output


def count_mentions(text: str):
    """Count the number of mentions in a tweet"""
    return text.count("@")


def count_hashtags(text: str):
    """Count the number of hashtags in a tweet"""
    return text.count("#")


def get_emoji_meaning(text: str):
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


class TwitterDataset(Dataset):
    """Dataset of tweets and corresponding hate-speech rating, toxicity and target label"""

    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len: int):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tokenizer.generate_vocab(self.df['c_text'].tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        text = row['c_text']
        rating = row['hatespeech']
        target = row['target']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_len=self.max_len,
            padding='max_length',
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'rating': torch.tensor(rating, dtype=torch.long),
            'target': torch.tensor(["person", "group", "public", ""].index(target), dtype=torch.long)
        }


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


def main(debug=False, use_k_fold=True):
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

    if use_k_fold:
        splits = cross_validation_split(df, n_splits=5)

        for train, test, i in splits:
            print('-' * 80)
            print(f'Split: {i}')
            print('Training...')
            X = np.array(
                [[count_mentions(text), get_emoji_meaning(text), count_hashtags(text)] for text in train['c_text']])
            Y = np.array(train['hatespeech'])
            clf = best[0]
            clf.fit(X, Y)
            train_score = clf.score(X, Y)
            print(f'Accuracy: {train_score}')

            # test with the test set
            print('Testing...')
            X = np.array(
                [[count_mentions(text), get_emoji_meaning(text), count_hashtags(text)] for text in test['c_text']])
            Y = np.array(test['hatespeech'])
            test_score = clf.score(X, Y)
            print(f'Accuracy: {test_score}')

            if test_score > best[1]:
                best[1] = test_score

        # print some example classifications
        if debug:
            print('-' * 80)
            clf = best[0]
            print('Example classifications')
            for text in df['c_text'].sample(10):
                print(
                    f'Tweet with {count_mentions(text)} mentions and {count_hashtags(text)} hashtags and emojis:{get_emoji_meaning(text)} should be {clf.predict([[count_mentions(text), get_emoji_meaning(text), count_hashtags(text)]])[0]}')
                print(f'Actual hate-speech rating: {df[df["c_text"] == text]["hatespeech"].values[0]}')

            dat = df.sample(1)
            X = np.array(
                [[count_mentions(text), get_emoji_meaning(text), count_hashtags(text)] for text in dat['c_text']])
            Y = np.array(dat['hatespeech'])
            metrics.ConfusionMatrixDisplay.from_estimator(clf, X, Y).plot()
            plt.show()
    else:
        train = df.sample(frac=0.8)
        test = df.drop(train.index)
        print('-' * 80)
        print('Training...')
        X = np.array(
            [[get_emoji_meaning(text), count_mentions(text)] for text in train['c_text']])
        Y = np.array(train['hatespeech'])
        cluster = best[0]
        cluster.fit(X, Y)
        clf = best[1]
        best.append(InductiveClusterer(cluster, clf))
        indl = best[2].fit(X, Y)
        # score is distance of prediction to actual value
        train_score = sum([abs(indl.predict(X)[i] - Y[i]) for i in range(len(Y))]) / len(Y)
        print(f'Accuracy: {train_score}')

        # test with the test set
        print('Testing...')
        X = np.array([[get_emoji_meaning(text), count_mentions(text)] for text in test['c_text']])
        Y = np.array(test['hatespeech'])
        test_score = sum([abs(indl.predict(X)[i] - Y[i]) for i in range(len(Y))]) / len(Y)
        print(f'Accuracy: {test_score}')

        # print some example classifications
        if debug:
            print('-' * 80)
            cluster = best[0]
            clf = best[1]
            indl = best[2].fit(X)
            print('Example classifications')
            for text in df['c_text'].sample(10):
                print(
                    f'Tweet with {count_mentions(text)} mentions and {count_hashtags(text)} hashtags and emojis:{get_emoji_meaning(text)} should be {indl.predict([[get_emoji_meaning(text), count_mentions(text)]])[0]}')
                print(f'Actual hate-speech rating: {df[df["c_text"] == text]["hatespeech"].values[0]}')

            dat = df.sample(300)
            X = np.array([[get_emoji_meaning(text), count_mentions(text)] for text in dat['c_text']])
            Y = np.array(dat['hatespeech'])
            metrics.ConfusionMatrixDisplay.from_estimator(clf, X, Y).plot()
            metrics.RocCurveDisplay.from_estimator(clf, X, Y).plot()
            metrics.PrecisionRecallDisplay.from_estimator(clf, X, Y).plot()
            plt.show()


if __name__ == '__main__':
    main(True, False)
