import pandas as pd
import torch
from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from torch import nn, tensor
from torch.utils.data import Dataset
from torch.nn import functional as F


def _classifier_has(attr):
    """Check if we can delegate a method to the underlying classifier.

    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    return lambda estimator: (
        hasattr(estimator.classifier_, attr)
        if hasattr(estimator, "classifier_")
        else hasattr(estimator.classifier, attr)
    )


class InductiveClusterer(BaseEstimator):
    def __init__(self, clusterer, classifier):
        self.classifier_ = None
        self.clusterer_ = None
        self.clusterer = clusterer
        self.classifier = classifier
        self.best_score_ = None

    def fit(self, x, y=None):
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y = self.clusterer_.fit_predict(x)
        self.classifier_.fit(x, y)
        return self

    def score(self, x, y):
        score = self.classifier_.score(x, y)
        if self.best_score_ is None or score > self.best_score_:
            self.best_score_ = score
        return score

    @available_if(_classifier_has("predict"))
    def predict(self, X):
        check_is_fitted(self)
        return self.classifier_.predict(X)

    @available_if(_classifier_has("decision_function"))
    def decision_function(self, X):
        check_is_fitted(self)
        return self.classifier_.decision_function(X)


class LNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


class LnnDataset(Dataset):
    """Loads a Dataframe of features and targets into a Dataset"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = self.df.iloc[idx]["features"]
        target = ["person", "group", "public"].index(self.df.iloc[idx]["target"])
        # convert target to one-hot
        target = tensor([1 if i == target else 0 for i in range(3)])
        return tensor(features), target
