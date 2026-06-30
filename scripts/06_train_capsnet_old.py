#!/usr/bin/env python3
"""Train a CapsNet classifier on Gene-PCA features.

This implementation follows the DiseaseCapsule-style architecture used in the manuscript. The original
DiseaseCapsule repository is available at https://github.com/HaploKit/DiseaseCapsule.
"""

import argparse
import os
import pickle
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_idx(path):
    with open(path) as handle:
        return np.array([int(x.strip()) for x in handle if x.strip()], dtype=int)


class ConvCaps2D(nn.Module):
    def __init__(self, primary_capslen, filters, kernel_size, stride):
        super().__init__()
        self.primary_capslen = primary_capslen
        self.capsules = nn.ModuleList([
            nn.Conv2d(1, primary_capslen, kernel_size=(1, kernel_size), stride=stride)
            for _ in range(filters)
        ])

    @staticmethod
    def squash(tensor, dim=-1):
        norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        return (norm / (1 + norm)) * tensor / torch.sqrt(norm + 1e-8)

    def forward(self, x):
        outputs = [capsule(x).view(x.size(0), self.primary_capslen, -1) for capsule in self.capsules]
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return self.squash(outputs)


class Caps1D(nn.Module):
    def __init__(self, num_routes, primary_capslen, digital_capslen, num_iterations=3, num_caps=2):
        super().__init__()
        self.num_iterations = num_iterations
        self.W = nn.Parameter(torch.randn(num_caps, num_routes, primary_capslen, digital_capslen) * 0.01)

    @staticmethod
    def squash(tensor, dim=-1):
        norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        return (norm / (1 + norm)) * tensor / torch.sqrt(norm + 1e-8)

    def forward(self, u):
        u_ji = torch.matmul(u[:, None, :, None, :], self.W)
        b = torch.zeros_like(u_ji)
        for i in range(self.num_iterations):
            c = F.softmax(b, dim=2)
            v = self.squash((c * u_ji).sum(dim=2, keepdim=True))
            if i != self.num_iterations - 1:
                b = b + (u_ji * v).sum(dim=-1, keepdim=True)
        v = v.squeeze()
        classes = torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)
        return F.softmax(classes, dim=1)


class CapsNet(nn.Module):
    def __init__(self, n_features, neurons=150, dropout=0.5, primary_capslen=4,
                 digital_capslen=16, kernel_size=5, stride=2, filters=32, num_iterations=3):
        super().__init__()
        self.fc1 = nn.Linear(n_features, neurons)
        self.dropout = nn.Dropout(dropout)
        self.primary = ConvCaps2D(primary_capslen, filters, kernel_size, stride)
        num_routes = (int((neurons - kernel_size) / stride) + 1) * filters
        self.digit = Caps1D(num_routes, primary_capslen, digital_capslen, num_iterations)

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        x = x.reshape(x.shape[0], 1, 1, x.shape[1])
        x = self.primary(x)
        return self.digit(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path, help="Pickle containing (X dataframe/array, y).")
    parser.add_argument("--train-idx", required=True, type=Path)
    parser.add_argument("--test-idx", required=True, type=Path)
    parser.add_argument("--outdir", default="capsnet_results", type=Path)
    parser.add_argument("--seed", type=int, default=1521024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=1.0)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with args.features.open("rb") as handle:
        X, y = pickle.load(handle)
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    train_idx = read_idx(args.train_idx)
    test_idx = read_idx(args.test_idx)
    if args.train_ratio < 1.0:
        n = int(len(train_idx) * args.train_ratio)
        train_idx = np.random.default_rng(args.seed).choice(train_idx, n, replace=False)

    x_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    x_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.long)

    loaders = {
        "train": DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True),
        "test": DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False),
    }

    model = CapsNet(n_features=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        scheduler.step()
        train_loss = total_loss / len(loaders["train"].dataset)
        history.append({"epoch": epoch + 1, "train_loss": train_loss})
        print(f"epoch={epoch+1} train_loss={train_loss:.5f}")

    model.eval()
    probs = []
    true = []
    with torch.no_grad():
        for xb, yb in loaders["test"]:
            out = model(xb.to(device)).cpu().numpy()
            probs.append(out[:, 1])
            true.append(yb.numpy())
    probs = np.concatenate(probs)
    true = np.concatenate(true)
    pred = (probs >= 0.5).astype(int)

    acc = accuracy_score(true, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average="binary", zero_division=0)
    auc = roc_auc_score(true, probs)
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    metrics = pd.DataFrame([{
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "auc": auc, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }])
    metrics.to_csv(args.outdir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.outdir / "training_history.csv", index=False)
    torch.save(model.state_dict(), args.outdir / "capsnet_model.pt")

    fpr, tpr, _ = roc_curve(true, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.outdir / "roc_curve.png", dpi=300)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
