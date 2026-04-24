#!/usr/bin/env python3
"""Build per-gene genotype matrices from annotated VCFs.

Inputs
------
- chr*.variant_function.uniqGene files with at least three columns: chrom/pos, SNP_ID_or_pos, gene.
- VCF files with GT fields for the same SNPs.
- labels.tsv/csv with columns FID and Pheno, where Pheno is 1=control and 2=case.

Outputs
-------
For each gene, writes <outdir>/chr<chrom>/genes/<gene>.pkl containing (X, Y, snp_ids), where
X is samples x SNPs and Y is one-hot encoded labels.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

GENOTYPE_TO_DOSAGE = {
    "0/0": 0, "0|0": 0,
    "0/1": 1, "1/0": 1, "0|1": 1, "1|0": 1,
    "1/1": 2, "1|1": 2,
    "./.": -1, ".|.": -1, ".": -1,
}


def read_labels(path: Path):
    labels = pd.read_csv(path, sep=None, engine="python")
    if "Pheno" not in labels.columns:
        raise ValueError("Labels file must contain a 'Pheno' column with 1=control, 2=case.")
    y = np.array([[1, 0] if int(v) == 1 else [0, 1] for v in labels["Pheno"]], dtype=np.int64)
    return labels, y


def read_gene_map(path: Path):
    gene2snps = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 3:
                continue
            snp = fields[1]
            gene = fields[2].split(",")[0].split(";")[0]
            gene2snps.setdefault(gene, []).append(snp)
    return gene2snps


def read_vcf_genotypes(path: Path):
    snp2geno = {}
    samples = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip().split("\t")[9:]
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 10:
                continue
            snp_id = fields[2] if fields[2] != "." else fields[1]
            gt_values = []
            for sample_field in fields[9:]:
                gt = sample_field.split(":", 1)[0]
                gt_values.append(GENOTYPE_TO_DOSAGE.get(gt, -1))
            snp2geno[snp_id] = gt_values
            # Some legacy annotations use position rather than rsID.
            snp2geno[fields[1]] = gt_values
    return samples, snp2geno


def majority_impute(matrix: np.ndarray, missing_threshold: float):
    keep_cols = []
    imputed_cols = []
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        non_missing = col[col >= 0]
        if len(non_missing) / len(col) < (1 - missing_threshold):
            continue
        values, counts = np.unique(non_missing, return_counts=True)
        majority = values[np.argmax(counts)] if len(values) else 0
        col = np.where(col < 0, majority, col)
        keep_cols.append(j)
        imputed_cols.append(col)
    if not imputed_cols:
        return np.empty((matrix.shape[0], 0), dtype=np.float32), []
    return np.vstack(imputed_cols).T.astype(np.float32), keep_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--outdir", default="prepared", type=Path)
    parser.add_argument("--missing-threshold", default=0.50, type=float, help="Maximum missingness allowed per SNP.")
    args = parser.parse_args()

    _, y = read_labels(args.labels)
    gene2snps = read_gene_map(args.annotation)
    samples, snp2geno = read_vcf_genotypes(args.vcf)
    if len(samples) != y.shape[0]:
        raise ValueError(f"VCF has {len(samples)} samples but labels file has {y.shape[0]} rows. Ensure identical order.")

    gene_dir = args.outdir / f"chr{args.chrom}" / "genes"
    gene_dir.mkdir(parents=True, exist_ok=True)

    n_genes = 0
    for gene, snps in gene2snps.items():
        rows, observed_snps = [], []
        for snp in snps:
            if snp in snp2geno:
                rows.append(snp2geno[snp])
                observed_snps.append(snp)
        if not rows:
            continue
        x = np.array(rows, dtype=np.float32).T
        x, kept = majority_impute(x, args.missing_threshold)
        if x.shape[1] == 0:
            continue
        kept_snps = [observed_snps[i] for i in kept]
        with (gene_dir / f"{gene}.pkl").open("wb") as out:
            pickle.dump((x, y, kept_snps), out)
        n_genes += 1
    print(f"Wrote {n_genes} gene matrices to {gene_dir}")


if __name__ == "__main__":
    main()
