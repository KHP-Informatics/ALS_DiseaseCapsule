#! /bin/bash

pval=p_0.05
for gwas in lmm meta
do
cd /scratch/users/k20066216/vcf_imputed
for file in sFR sBE sCZ sFIN sGER sIR sSW sIB sUS sUK sIT sNL 
do
/scratch/users/k20066216/plink  --bfile /scratch/users/k20066216/vcf_imputed/$file --extract /scratch/prj/bcn_ml_als/DiseaseCapsule/GWAS_2016/$pval/$gwas/sig_snps.txt  --threads 16 --make-bed --out  /scratch/prj/bcn_ml_als/DiseaseCapsule/GWAS_2016/$pval/$gwas/$file.sig
done
done

