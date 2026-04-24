#!/usr/bin/env python3
"""Run Gene-PCA for each gene matrix.

The PCA is fitted on training samples only, then applied to all samples. This avoids leakage from
validation/test data while preserving the DiseaseCapsule Gene-PCA feature representation.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA


def read_idx(path):
    if path is None:
        return None
    with open(path) as handle:
        return np.array([int(x.strip()) for x in handle if x.strip()], dtype=int)


def n_components(n_snps):
    if n_snps > 20:
        return 8
    if n_snps > 4:
        return 4
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes-dir", required=True, type=Path, help="Directory containing gene .pkl files.")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--train-idx", type=Path, default=None, help="Optional newline-delimited training indices. If omitted, fits PCA on all samples.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    train_idx = read_idx(args.train_idx)

    for gene_file in sorted(args.genes_dir.glob("*.pkl")):
        with gene_file.open("rb") as handle:
            loaded = pickle.load(handle)
        x = loaded[0]
        if x.shape[1] == 0:
            continue
        k = min(n_components(x.shape[1]), x.shape[0], x.shape[1])
        fit_x = x if train_idx is None else x[train_idx]
        pca = PCA(n_components=k)
        pca.fit(fit_x)
        embedding = pca.transform(x)
        with (args.outdir / gene_file.name).open("wb") as out:
            pickle.dump({"embedding": embedding, "model": pca, "gene": gene_file.stem}, out)
        print(gene_file.stem, x.shape, "->", embedding.shape, "variance", pca.explained_variance_ratio_.sum())


if __name__ == "__main__":
    main()
