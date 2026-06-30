#!/usr/bin/env python3
"""Project external per-gene genotype matrices into trained Gene-PCA spaces.

This script applies the PCA models fitted by `03_gene_pca.py` to external samples prepared by
`get_genes.py`. It does not refit PCA on the external samples, preventing information leakage and
allowing one or more external samples to be processed without retraining the full model.

Inputs
------
- External gene matrices: <external-genes-dir>/<GENE>.pkl, containing (X, None, snp_ids).
- Training Gene-PCA model files from `03_gene_pca.py`: <training-pca-dir>/<GENE>.pkl.

Outputs
-------
- <outdir>/<GENE>.pkl dictionaries with keys: embedding, gene.
These files are compatible with `04_combine_gene_pcs.py`.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def load_external_x(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, dict):
        x = obj.get("X", None)
        if x is None:
            x = obj.get("x", None)
        if x is None:
            x = obj.get("matrix", None)
        if x is None:
            raise ValueError(f"No genotype matrix found in {path}")
    elif isinstance(obj, tuple):
        x = obj[0]
    else:
        raise ValueError(f"Unsupported external gene pickle format in {path}")
    return np.asarray(x, dtype=np.float32)


def load_pca_model(path: Path):
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    raise ValueError(
        f"{path} does not contain a saved PCA model. Re-run scripts/03_gene_pca.py, which saves both embeddings and PCA models."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-genes-dir", required=True, type=Path,
                        help="Directory containing external gene matrices from get_genes.py, e.g. external_prepared/chr1/genes.")
    parser.add_argument("--training-pca-dir", required=True, type=Path,
                        help="Directory containing fitted training PCA files from 03_gene_pca.py, e.g. prepared/chr1/pca.")
    parser.add_argument("--outdir", required=True, type=Path,
                        help="Output directory, e.g. external_prepared/chr1/pca.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if an external gene has no matching PCA model. By default, unmatched genes are skipped.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0

    for external_gene_file in sorted(args.external_genes_dir.glob("*.pkl")):
        gene = external_gene_file.stem
        pca_file = args.training_pca_dir / external_gene_file.name
        if not pca_file.exists():
            if args.strict:
                raise FileNotFoundError(f"No trained PCA model found for {gene}: {pca_file}")
            n_skipped += 1
            continue
        x_external = load_external_x(external_gene_file)
        pca_model = load_pca_model(pca_file)
        embedding = pca_model.transform(x_external)
        with (args.outdir / external_gene_file.name).open("wb") as out:
            pickle.dump({"embedding": embedding, "gene": gene}, out)
        print(gene, x_external.shape, "->", embedding.shape)
        n_written += 1

    print(f"Projected {n_written} genes to {args.outdir}; skipped {n_skipped} genes without PCA models.")


if __name__ == "__main__":
    main()
