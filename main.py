import math
import re
import time

import emoji
import torch
import nltk
import torch.nn as nn
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class TweetDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, tokenizer):
        self.dataset = dataset
        self.processor = tokenizer
        self.len = len(dataset)

    def __getitem__(self, index):
        return self.processor({'c_text': self.dataset.iloc[index, 1]}), self.dataset.iloc[index, 9]

    def __len__(self):
        return self.len


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TweetTransformer(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def yield_tokens(data_iter, tokenizer, do_filtering=False, only_hashtags=False):
    for text in data_iter['c_text']:
        if do_filtering:
            text = ' '.join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
            text = re.sub(r'http\S+', ' ', text)
        if only_hashtags:
            text = ' '.join(re.findall(r"#(\w+)", text))
        yield tokenizer.tokenize(text)


def importDataset(tsvfile):
    # Import
    dataset = pd.read_csv(tsvfile, sep='\t', header=0)
    return dataset


def printDSInfo(dataset):
    # Print info
    print(dataset.info())
    print(dataset.head())


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def data_process(raw_text_iter):
        tokens = [i for i in yield_tokens(raw_text_iter, tokenizer, True)]
        data = [torch.tensor(vocab(token), dtype=torch.long) for token in tokens]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    # Import dataset
    dataset = importDataset('data.tsv')
    ds = TweetDataset(dataset, data_process)

    # print some info about the dataset
    # printDSInfo(ds.dataset)

    # TODO Implement own tokenizer
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    # tokenize the first entry of the dataset
    vocab = build_vocab_from_iterator(yield_tokens(ds.dataset, tokenizer, False, True), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    print(vocab.get_itos())
    print(len(vocab.get_itos()))

    # create a dataloader


if __name__ == '__main__':
    main()
