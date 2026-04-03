#!/usr/bin/env python

# -------- markdown --------
# ## import packages

import pandas as pd
import numpy as np
from itertools import product
import os
import sys
import random
import copy
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#import umap

#chrom='22'
chrom=sys.argv[1]
#file=sys.argv[2]
file = 'pm.sig'
total_snps = 0
snps_missingness_less_than_50 = 0
'''
# gene to strand
gene2strand={}
with open('gene_strand.txt') as fr:
    for line in fr:
        gene,strand=line.strip().split()
        gene2strand[gene]=strand
'''
# read gene to snp file
gene2snp={}
# corrdinates of snps should be sorted
with open('chr' + str(chrom) +".variant_function.uniqGene",'r') as fr:
    for line in fr:
        chromosome,snp,gene=line.split()[:3]
        if gene in gene2snp:
            gene2snp[gene].append(snp)
        else:
            gene2snp[gene]=[snp]


len(gene2snp)


vcf = str(file) + '.'+ str(chrom) +'.vcf'
snp2genotype={}
with open(vcf) as fr:
    for line in fr:
        if line.startswith("##"):
            continue
        elif line.startswith("#CHROM"):
            header=line.strip().split()[9:]
        else:
            a=line.strip().split()
            # snp2genotype[a[1]]='\t'.join(a[9:]) # keys: '17043379', '17052123'
            snp2genotype[a[1]]='\t'.join(a[9:])


labels_file='pm_labels.csv'
labels_df = pd.read_csv(labels_file, index_col=0,sep='\t')
sample_ids = labels_df.FID.tolist()
sample_ids[:4]

# the output is a vector for probability in DL
lab_num = {1: [1, 0], # negative, max_idx=0
           2: [0, 1]} # positive, max_idx=1
pheno_new = []
for i in labels_df.Pheno.tolist():
    pheno_new.append(lab_num[i])


# mapping dictionary
genotype2num = {"0/0":'0',
                "0/1":'1',
                "1/0":'1',
                "1/1":'2',
                "./.":'-1'}

os.system("mkdir -p chr" +  str(chrom))
os.system("mkdir -p chr" +  str(chrom) + "/pca")
os.system("mkdir -p chr" +  str(chrom) + "/genes")

def impute_missing_majority(snp_data):
    # Calculate the counts of each allele
    allele_counts = {0: 0, 1: 0, 2: 0, -1: 0}
    for geno in snp_data:
        if geno == -1:
            continue
        allele_counts[geno] += 1
    # Find the majority allele if more than 50% non-missing genotypes exist
    non_missing_count = allele_counts[0] + allele_counts[1] + allele_counts[2]
    if non_missing_count / len(snp_data) >= 0.5:
        majority_allele = max(allele_counts, key=allele_counts.get)
        # Impute missing data with the majority allele
        imputed_data = [majority_allele if geno == -1 else geno for geno in snp_data]
        return majority_allele
    else:
        return -1
    
def count_snps_missingness_less_than_50(genotype_data_array, threshold=0.5):
    missing_counts = np.sum(genotype_data_array == -1, axis=0)  # Count occurrences of -1 (missing genotypes)
    selected_snps = missing_counts <= threshold * len(genotype_data_array)
    return np.sum(selected_snps)

for gene in gene2snp.keys(): # 'ACR','CCT8L2'
    #     gene='LINC00308'
    print('processing gene: {}'.format(gene))
    genotypes = []
    for snp in gene2snp[gene]: # gene2snp[gene]: 'ACR': ['22:51177257', '22:51178065', '22:51178090', '22:51178607']
        try:
            genotypes.append(snp2genotype[snp]) # snp2genotype.keys(): ['rs114553188', 'rs375798137', 'rs9616985', 'rs376461333', 'rs3896457']
        except KeyError:
            print('Skipping SNP: {}'.format(snp))
            continue
    genotypes2=[]
    for genotype in genotypes:
        genotypes2.append([genotype2num[geno] for geno in genotype.split()])

    dataset_X = np.array(genotypes2).astype('float32').T
    #dataset_X[dataset_X == -1] = np.nan
    dataset_X = dataset_X.reshape((len(dataset_X), dataset_X.shape[1]))
    total_snps += dataset_X.shape[1]
    majority_allele = np.apply_along_axis(impute_missing_majority, 0, dataset_X)
    majority_alleles = {}
    for i, snp in enumerate(gene2snp[gene]):
        if majority_allele[i] is not None:
            majority_alleles[snp] = majority_allele[i]

    # Write majority alleles to a text file
    output_file = "chr" +  str(chrom) + "/" + gene +  "_majority_alleles.txt" # Adjust the filename as needed
    with open(output_file, "w") as fw:
        for snp, allele in majority_alleles.items():
            fw.write("{}\t{}\n".format(snp, allele))

# Print the results
print("Total SNPs:", total_snps)
print("SNPs with missingness less than 50%:", snps_missingness_less_than_50)
print("SNPs with missingness greater than or equal to 50%:", total_snps - snps_missingness_less_than_50)


