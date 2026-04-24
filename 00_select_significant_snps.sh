#!/usr/bin/env bash
set -euo pipefail

# Select SNPs below a chosen GWAS p-value threshold and extract them from PLINK files.
# This is a template: edit paths/strata for your local data.

GWAS_SUMSTATS_DIR=${GWAS_SUMSTATS_DIR:-summary_statistics}
PLINK_BIN=${PLINK_BIN:-plink}
PLINK_INPUT_DIR=${PLINK_INPUT_DIR:-plink_bfiles}
OUTDIR=${OUTDIR:-results/p_0.05/meta}
P_THRESHOLD=${P_THRESHOLD:-0.05}
GWAS_TYPE=${GWAS_TYPE:-meta}
STRATA=${STRATA:-"sFR sBE sCZ sFIN sGER sIR sSW sIB sUS sUK sIT sNL"}

mkdir -p "$OUTDIR"
: > "$OUTDIR/sig_snps.txt"

for chr in {1..22}; do
  awk -v p="$P_THRESHOLD" '{ if ($6 >= 0.01 && $9 <= p) print $2 }' \
    "$GWAS_SUMSTATS_DIR/als.sumstats.${GWAS_TYPE}.chr${chr}.txt" >> "$OUTDIR/sig_snps.txt"
done

sort -u "$OUTDIR/sig_snps.txt" -o "$OUTDIR/sig_snps.txt"
wc -l "$OUTDIR/sig_snps.txt"

for cohort in $STRATA; do
  "$PLINK_BIN" \
    --bfile "$PLINK_INPUT_DIR/$cohort" \
    --extract "$OUTDIR/sig_snps.txt" \
    --threads ${THREADS:-8} \
    --make-bed \
    --out "$OUTDIR/${cohort}.sig"
done
