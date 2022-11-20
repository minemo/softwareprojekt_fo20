import pandas as pd
import numpy as np
import torch
import nltk
import demoji
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import TwitterHatespeechModel


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
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
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
        splits.append((train, test))
    return splits


def main(debug=False):
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
        print('-' * 80)

    # create a Dataloader with k-fold cross validation
    splits = cross_validation_split(df, 5)

    for i, (train, test) in enumerate(splits):
        train_dataset = TwitterDataset(train, TweetTokenizer(), 50)
        test_dataset = TwitterDataset(test, TweetTokenizer(), 50)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        if debug:
            print(f"Fold {i + 1}")
            print(f"Train dataset: {len(train_dataset)}")
            print(f"Test dataset: {len(test_dataset)}")
            print(f"Train dataloader: {len(train_dataloader)}")
            print(f"Test dataloader: {len(test_dataloader)}")
            print('-' * 80)


if __name__ == '__main__':
    main(True)
