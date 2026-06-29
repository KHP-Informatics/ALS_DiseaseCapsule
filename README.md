# ALS DiseaseCapsule implementation

This repository contains the analysis code used for the manuscript **"Towards a deep-learning genomic tool for risk stratification and diagnostic support in sporadic ALS"**.

The trained CapsuleNet weights of the model trained on the full 2016 ALS GWAS data can be found in `capsule_pca_1.0model_1.pt`   

The work builds on the published DiseaseCapsule framework and architecture from Luo et al. The original DiseaseCapsule code is available here:

<https://github.com/HaploKit/DiseaseCapsule>

This repository provides the additional scripts used in our ALS implementation, including SNP selection, gene-level annotation, majority imputation of missing genotypes, Gene-PCA feature generation, train/test splitting, and CapsNet model training/evaluation.

## Important data-access note

Raw genotype data from Project MinE / ALS GWAS cohorts are **not included** in this repository because they are controlled-access human genetic data. The scripts therefore require users to provide their own genotype data in PLINK/VCF format, together with phenotype labels in the same sample order.

The scripts are intended to make the computational workflow transparent and reproducible for users with appropriate access to equivalent data.

## Repository structure

```text
ALS_DiseaseCapsule/
├── README.md
├── environment.yml
├── capsule_pca_1.0model_1.pt
├── config/
│   └── example_config.yaml
├── data/
│   └── README_inputs.md
├── scripts/
│   ├── 00_select_significant_snps.sh
│   ├── 01_annotate_snps_annovar.sh
│   ├── 02_build_gene_matrices.py
│   ├── 03_gene_pca.py
│   ├── 04_combine_gene_pcs.py
│   ├── 05_make_splits.py
│   └── 06_train_capsnet.py
└── legacy_scripts/
    └── Original scripts/logs used internally for the manuscript analyses
```

## Overview of the workflow

The pipeline follows the main steps described in the manuscript:

1. Select SNPs using GWAS summary statistics, using the same threshold as DiseaseCapsule (`p < 0.05`).
2. Extract selected SNPs from cohort-level PLINK files.
3. Convert/extract selected SNPs to VCF and annotate SNPs to genes using ANNOVAR.
4. Build one genotype matrix per gene.
5. Replace missing genotypes by the majority genotype at each SNP.
6. Run Gene-PCA per gene. PCA is fitted on training samples only and then applied to the validation/test samples.
7. Combine all gene-level PCs into a single feature matrix.
8. Create balanced, leave-cohort-out, or population-screening train/test indices.
9. Train and evaluate the CapsNet classifier.

## Requirements

The cleaned scripts require Python 3.9+ and the software version and packages listed in `environment.yml`.

Create the environment with:

```bash
conda env create -f environment.yml
conda activate als-diseasecapsule
```

External command-line tools used in the full workflow:

- PLINK 1.9 or 2.0
- ANNOVAR
- a VCF/BCF manipulation tool such as bcftools, if required by your local preprocessing workflow

## Expected input files

See `data/README_inputs.md` for full details. In brief, the workflow expects:

- GWAS summary statistics, split by chromosome, with columns including SNP ID, MAF and p-value.
- PLINK binary files (`.bed/.bim/.fam`) or VCF files for genotype data.
- A phenotype/label file containing one row per sample and at least:
  - `FID`: sample identifier
  - `Pheno`: phenotype code, where `1 = control` and `2 = ALS case`
- ANNOVAR gene annotation files generated from selected SNP VCFs.

The phenotype file must be in the same sample order as the VCF files used to build gene matrices.

## Step-by-step usage

### 1. Select GWAS-significant SNPs and extract them from PLINK files

Edit paths in `scripts/00_select_significant_snps.sh` or set them as environment variables:

```bash
export GWAS_SUMSTATS_DIR=/path/to/summary_statistics
export PLINK_INPUT_DIR=/path/to/plink_bfiles
export PLINK_BIN=/path/to/plink
export OUTDIR=results/p_0.05/meta
export P_THRESHOLD=0.05
bash scripts/00_select_significant_snps.sh
```

This produces:

```text
results/p_0.05/meta/sig_snps.txt
results/p_0.05/meta/<cohort>.sig.{bed,bim,fam}
```

### 2. Annotate selected SNPs to genes

First create per-chromosome VCFs from the selected SNP set using your preferred local workflow. Then run:

```bash
export ANNOVAR_DIR=/path/to/annovar
export HUMANDB=/path/to/annovar/humandb
export VCF_PREFIX=all.sig
export OUTDIR=annotations
bash scripts/01_annotate_snps_annovar.sh
```

This produces files such as:

```text
annotations/chr1.variant_function.uniqGene
annotations/chr2.variant_function.uniqGene
...
```

### 3. Build per-gene genotype matrices

For each chromosome:

```bash
python scripts/02_build_gene_matrices.py \
  --chrom 1 \
  --vcf all.sig.1.vcf \
  --annotation annotations/chr1.variant_function.uniqGene \
  --labels labels.csv \
  --outdir prepared
```

Output:

```text
prepared/chr1/genes/<GENE>.pkl
```

Each gene pickle contains:

```python
(X, Y, snp_ids)
```

where `X` is a samples-by-SNPs genotype dosage matrix and `Y` is a one-hot encoded phenotype matrix.

### 4. Create train/test splits

For a balanced test set:

```bash
python scripts/05_make_splits.py \
  --labels labels.csv \
  --outdir splits/balanced \
  --mode balanced \
  --test-fraction 0.10
```

For the population-screening simulation used in the manuscript:

```bash
python scripts/05_make_splits.py \
  --labels labels.csv \
  --outdir splits/population_1 \
  --mode population \
  --population-cases 30 \
  --population-controls 9000 \
  --seed 1
```

The script writes:

```text
splits/.../test.idx
splits/.../train_val.unique.idx
splits/.../train_val.balanced.idx
```

### 5. Run Gene-PCA

Run this for each chromosome directory. PCA is fitted on training samples only and then applied to all samples.

```bash
python scripts/03_gene_pca.py \
  --genes-dir prepared/chr1/genes \
  --outdir prepared/chr1/pca \
  --train-idx splits/balanced/train_val.unique.idx
```

Output:

```text
prepared/chr1/pca/<GENE>.pkl
```

### 6. Combine Gene-PCA features

```bash
python scripts/04_combine_gene_pcs.py \
  --pca-root prepared \
  --labels labels.csv \
  --output sigSNPs_pca.features.pkl
```

This produces the final feature matrix used by the model:

```text
sigSNPs_pca.features.pkl
```

### 7. Train and evaluate CapsNet

```bash
python scripts/06_train_capsnet.py \
  --features sigSNPs_pca.features.pkl \
  --train-idx splits/balanced/train_val.balanced.idx \
  --test-idx splits/balanced/test.idx \
  --outdir results/capsnet_balanced \
  --epochs 30 \
  --batch-size 128
```

Output:

```text
results/capsnet_balanced/metrics.csv
results/capsnet_balanced/training_history.csv
results/capsnet_balanced/capsnet_model.pt
results/capsnet_balanced/roc_curve.png
```

## Leave-one-out and leave-half-out experiments, and external validation

The manuscript used national cohort-based splits. These can be reproduced by creating index files corresponding to the held-out national cohort or half-cohort, then passing those index files to the same Gene-PCA and model-training scripts.

For strict external validation, make sure that:

1. SNP selection is performed using training GWAS summary statistics only.
2. PCA is fitted on training data only.
3. External samples are projected onto the training-derived Gene-PCA space.
4. No external/test samples are used for hyperparameter tuning.

## Relationship to the original DiseaseCapsule repository

The deep learning architecture in `scripts/06_train_capsnet.py` follows the DiseaseCapsule-style CapsNet used in the manuscript. The original DiseaseCapsule implementation from Luo et al. should be cited and is available at:

<https://github.com/HaploKit/DiseaseCapsule>

## Legacy scripts

The `legacy_scripts/` directory contains the original working scripts and log file used internally to run the analyses on our HPC infrastructure. These are retained for transparency but include local paths and cluster-specific assumptions. Users should normally use the cleaned scripts in `scripts/`.

## Citation

If you use this code, please cite the manuscript **J Hu, et al. "Towards a deep-learning genomic tool for risk stratification and diagnostic support in sporadic ALS", medRxiv, 2025** and the original DiseaseCapsule paper.

## Disclaimer

This is research code. It is provided to support reproducibility of the analyses described in the manuscript and is not intended for clinical use.
