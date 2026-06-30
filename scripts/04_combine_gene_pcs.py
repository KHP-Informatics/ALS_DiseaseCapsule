#!/usr/bin/env python3
"""Combine all Gene-PCA embeddings into one feature matrix.

If labels are provided, the output pickle contains (X, y). If labels are omitted, the output
contains (X, None), which is useful for external-sample inference.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-root", required=True, type=Path, help="Root containing chr*/pca/*.pkl files.")
    parser.add_argument("--labels", type=Path, default=None, help="Optional labels file with Pheno column, same sample order as features.")
    parser.add_argument("--output", default="sigSNPs_pca.features.pkl", type=Path)
    args = parser.parse_args()

    matrices = []
    columns = []
    for pca_file in sorted(args.pca_root.glob("chr*/pca/*.pkl")):
        with pca_file.open("rb") as handle:
            obj = pickle.load(handle)
        emb = obj["embedding"] if isinstance(obj, dict) else obj
        gene_name = pca_file.stem
        chrom = pca_file.parent.parent.name
        matrices.append(emb)
        columns.extend([f"{chrom}:{gene_name}:PC{i+1}" for i in range(emb.shape[1])])

    if not matrices:
        raise RuntimeError("No PCA files found. Expected chr*/pca/*.pkl")
    X = pd.DataFrame(np.concatenate(matrices, axis=1), columns=columns)
    y = None
    if args.labels is not None:
        labels = pd.read_csv(args.labels, sep=None, engine="python")
        y = np.array([0 if int(v) == 1 else 1 for v in labels["Pheno"]], dtype=np.int64)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Feature matrix and labels have different numbers of samples.")
    with args.output.open("wb") as out:
        pickle.dump((X, y), out)
    label_text = "with labels" if y is not None else "without labels"
    print(f"Saved {X.shape[0]} samples x {X.shape[1]} Gene-PCs to {args.output} ({label_text})")


if __name__ == "__main__":
    main()
