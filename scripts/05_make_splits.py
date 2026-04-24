#!/usr/bin/env python3
"""Create train/test indices for balanced, cohort-held-out, or population-screening experiments."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def write_idx(path, idx):
    with open(path, "w") as out:
        out.write("\n".join(map(str, idx)) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path, help="Labels file with Pheno column: 1=control, 2=case.")
    parser.add_argument("--outdir", default="splits", type=Path)
    parser.add_argument("--mode", choices=["balanced", "population"], default="balanced")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--population-cases", type=int, default=30)
    parser.add_argument("--population-controls", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=1991)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    labels = pd.read_csv(args.labels, sep=None, engine="python")
    case_idx = labels.index[labels["Pheno"] == 2].to_numpy()
    ctrl_idx = labels.index[labels["Pheno"] == 1].to_numpy()

    if args.mode == "population":
        test_cases = rng.choice(case_idx, args.population_cases, replace=False)
        test_ctrls = rng.choice(ctrl_idx, args.population_controls, replace=False)
    else:
        n_test_each = int(min(len(case_idx), len(ctrl_idx)) * args.test_fraction)
        test_cases = rng.choice(case_idx, n_test_each, replace=False)
        test_ctrls = rng.choice(ctrl_idx, n_test_each, replace=False)

    test_idx = np.concatenate([test_cases, test_ctrls])
    rng.shuffle(test_idx)
    train_unique = np.array([i for i in labels.index if i not in set(test_idx)], dtype=int)

    train_cases = np.array([i for i in train_unique if labels.loc[i, "Pheno"] == 2], dtype=int)
    train_ctrls = np.array([i for i in train_unique if labels.loc[i, "Pheno"] == 1], dtype=int)
    if len(train_cases) < len(train_ctrls):
        extra_cases = rng.choice(train_cases, len(train_ctrls) - len(train_cases), replace=True)
        train_balanced = np.concatenate([train_unique, extra_cases])
    else:
        extra_ctrls = rng.choice(train_ctrls, len(train_cases) - len(train_ctrls), replace=True)
        train_balanced = np.concatenate([train_unique, extra_ctrls])
    rng.shuffle(train_balanced)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_idx(args.outdir / "test.idx", test_idx)
    write_idx(args.outdir / "train_val.unique.idx", train_unique)
    write_idx(args.outdir / "train_val.balanced.idx", train_balanced)
    print(f"Wrote split files to {args.outdir}")
    print(f"Test: {len(test_cases)} cases, {len(test_ctrls)} controls")
    print(f"Train unique: {len(train_unique)}; train balanced/resampled: {len(train_balanced)}")


if __name__ == "__main__":
    main()
