#!/usr/bin/env python3
"""Build external-sample per-gene genotype matrices using training-set SNP order and imputation.

This script prepares new/external samples for DiseaseCapsule inference. For each gene, it uses the
SNP list and order stored in the training gene matrices produced by `02_build_gene_matrices.py`.
Missing genotypes within the external VCF, and SNPs absent from the external VCF, are imputed with
the most common genotype observed for the corresponding SNP in the training data.

Inputs
------
- External per-chromosome VCF with GT fields.
- Training gene matrix directory, e.g. prepared/chr1/genes, containing <GENE>.pkl files.
- Optional annotation file. If provided, it can be used to restrict processing to genes annotated on
  the chromosome. The SNP order still comes from the training gene pickle.

Outputs
-------
- <outdir>/chr<chrom>/genes/<GENE>.pkl containing (X, None, snp_ids), where X is
  external samples x SNPs and snp_ids is the training SNP order.
- <outdir>/chr<chrom>/samples.txt containing the external VCF sample order.
- <outdir>/chr<chrom>/imputation_report.tsv summarising observed and imputed SNPs.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

GENOTYPE_TO_DOSAGE = {
    "0/0": 0, "0|0": 0,
    "0/1": 1, "1/0": 1, "0|1": 1, "1|0": 1,
    "1/1": 2, "1|1": 2,
    "./.": -1, ".|.": -1, ".": -1,
}


def read_gene_map(path: Optional[Path]) -> Optional[Dict[str, List[str]]]:
    if path is None:
        return None
    gene2snps: Dict[str, List[str]] = {}
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


def read_vcf_genotypes(path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    snp2geno: Dict[str, np.ndarray] = {}
    samples: List[str] = []
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
            gt_values = []
            for sample_field in fields[9:]:
                gt = sample_field.split(":", 1)[0]
                gt_values.append(GENOTYPE_TO_DOSAGE.get(gt, -1))
            values = np.asarray(gt_values, dtype=np.float32)
            # Support both rsID-based and position-based annotations.
            if fields[2] != ".":
                snp2geno[fields[2]] = values
            snp2geno[fields[1]] = values
            snp2geno[f"{fields[0]}:{fields[1]}"] = values
            snp2geno[f"chr{fields[0].replace('chr', '')}:{fields[1]}"] = values
    if not samples:
        raise ValueError(f"No #CHROM header with samples found in {path}")
    return samples, snp2geno


def unpack_training_gene_pickle(path: Path) -> Tuple[np.ndarray, Optional[Sequence[str]]]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, dict):
        x_value = obj.get("X", None)
        if x_value is None:
            x_value = obj.get("x", None)
        if x_value is None:
            x_value = obj.get("matrix", None)
        if x_value is None:
            raise ValueError(f"No genotype matrix found in {path}")
        x = np.asarray(x_value, dtype=np.float32)
        snp_ids = obj.get("snp_ids", None)
        if snp_ids is None:
            snp_ids = obj.get("snps", None)
    elif isinstance(obj, tuple):
        x = np.asarray(obj[0], dtype=np.float32)
        snp_ids = obj[2] if len(obj) >= 3 else None
    else:
        raise ValueError(f"Unsupported pickle format in {path}")
    return x, snp_ids


def majority_genotypes(x: np.ndarray, train_idx: Optional[np.ndarray] = None) -> np.ndarray:
    fit_x = x if train_idx is None else x[train_idx]
    majorities = []
    for j in range(fit_x.shape[1]):
        col = fit_x[:, j]
        non_missing = col[col >= 0]
        if non_missing.size == 0:
            majorities.append(0)
        else:
            values, counts = np.unique(non_missing.astype(int), return_counts=True)
            majorities.append(int(values[np.argmax(counts)]))
    return np.asarray(majorities, dtype=np.float32)


def read_idx(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    with path.open() as handle:
        return np.asarray([int(x.strip()) for x in handle if x.strip()], dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrom", required=True, help="Chromosome label, e.g. 1 or 22.")
    parser.add_argument("--vcf", required=True, type=Path, help="External-sample VCF for this chromosome.")
    parser.add_argument("--training-genes-dir", required=True, type=Path,
                        help="Training gene matrix directory, e.g. prepared/chr1/genes.")
    parser.add_argument("--annotation", type=Path, default=None,
                        help="Optional chr*.variant_function.uniqGene file to restrict genes processed.")
    parser.add_argument("--train-idx", type=Path, default=None,
                        help="Optional training indices used to estimate majority genotypes. If omitted, all training samples are used.")
    parser.add_argument("--outdir", default="external_prepared", type=Path,
                        help="Output root for external gene matrices.")
    args = parser.parse_args()

    train_idx = read_idx(args.train_idx)
    annotated_genes = set(read_gene_map(args.annotation).keys()) if args.annotation else None
    samples, snp2geno = read_vcf_genotypes(args.vcf)

    chrom_dir = args.outdir / f"chr{args.chrom}"
    gene_outdir = chrom_dir / "genes"
    gene_outdir.mkdir(parents=True, exist_ok=True)
    (chrom_dir / "samples.txt").write_text("\n".join(samples) + "\n")

    report_rows = []
    n_written = 0
    for train_gene_file in sorted(args.training_genes_dir.glob("*.pkl")):
        gene = train_gene_file.stem
        if annotated_genes is not None and gene not in annotated_genes:
            continue
        train_x, snp_ids = unpack_training_gene_pickle(train_gene_file)
        if snp_ids is None:
            raise ValueError(
                f"{train_gene_file} does not contain SNP IDs. Recreate training gene matrices with scripts/02_build_gene_matrices.py."
            )
        snp_ids = list(snp_ids)
        if train_x.shape[1] != len(snp_ids):
            raise ValueError(f"SNP ID count does not match matrix columns for {train_gene_file}")
        majorities = majority_genotypes(train_x, train_idx)

        cols = []
        n_present = 0
        n_missing_genotypes = 0
        n_absent_snps = 0
        for snp, majority in zip(snp_ids, majorities):
            if snp in snp2geno:
                col = snp2geno[snp].copy()
                n_present += 1
                n_missing_genotypes += int(np.sum(col < 0))
                col = np.where(col < 0, majority, col)
            else:
                col = np.full(len(samples), majority, dtype=np.float32)
                n_absent_snps += 1
            cols.append(col)

        if not cols:
            continue
        x_external = np.vstack(cols).T.astype(np.float32)
        with (gene_outdir / f"{gene}.pkl").open("wb") as out:
            pickle.dump((x_external, None, snp_ids), out)
        report_rows.append({
            "gene": gene,
            "n_snps": len(snp_ids),
            "snps_present_in_external_vcf": n_present,
            "snps_absent_imputed_from_training_majority": n_absent_snps,
            "missing_external_genotypes_imputed": n_missing_genotypes,
        })
        n_written += 1

    pd.DataFrame(report_rows).to_csv(chrom_dir / "imputation_report.tsv", sep="\t", index=False)
    print(f"Wrote {n_written} external gene matrices to {gene_outdir}")
    print(f"External sample order written to {chrom_dir / 'samples.txt'}")


if __name__ == "__main__":
    main()
