#!/usr/bin/env bash
set -euo pipefail

# Annotate per-chromosome VCF files with ANNOVAR and produce chr*.variant_function.uniqGene.
# Required inputs: chr${CHR}.vcf files containing selected/imputed SNPs.

ANNOVAR_DIR=${ANNOVAR_DIR:-/path/to/annovar}
HUMANDB=${HUMANDB:-/path/to/annovar/humandb}
VCF_PREFIX=${VCF_PREFIX:-all.sig}
OUTDIR=${OUTDIR:-annotations}
BUILD=${BUILD:-hg19}

mkdir -p "$OUTDIR"

for chr in ${CHROMS:-$(seq 1 22)}; do
  perl "$ANNOVAR_DIR/convert2annovar.pl" -format vcf4old "${VCF_PREFIX}.${chr}.vcf" > "$OUTDIR/chr${chr}.avinput" 2>/dev/null
  perl "$ANNOVAR_DIR/annotate_variation.pl" -out "$OUTDIR/chr${chr}" -build "$BUILD" "$OUTDIR/chr${chr}.avinput" "$HUMANDB"

  # Collapse ANNOVAR output to: chromosome position nearest_or_annotated_gene functional_class
  awk 'BEGIN{OFS="\t"} {print $3,$4,$2,$1}' "$OUTDIR/chr${chr}.variant_function" > "$OUTDIR/chr${chr}.variant_function.uniqGene"
done
