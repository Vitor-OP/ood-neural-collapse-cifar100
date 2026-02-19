import pandas as pd
import torch


def save_features_to_csv(features, labels, filepath):
    n_features = features.shape[1]
    feature_cols = [f"ct{i}" for i in range(n_features)]

    df = pd.DataFrame(features.numpy(), columns=feature_cols)
    df["label"] = labels.numpy()
    df.to_csv(filepath)


def load_features_from_csv(filepath):
    df = pd.read_csv(filepath, index_col=0)
    labels = df["label"].values

    feature_cols = [f"ct{i}" for i in range(64)]
    features = df.loc[:, feature_cols].values

    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
