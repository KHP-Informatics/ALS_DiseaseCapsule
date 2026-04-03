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
gwas='meta'
#gwas=sys.argv[2]
#file=sys.argv[2]
file='all.sig'
total_snps = 0
snps_missingness_less_than_50 = 0
'''
# gene to strand
gene2strand={}
with open('gene_strand.txt') as fr:
    for line in fr:
        gene,strand=line.strip().split()
        gene2strand[gene]=strand

# read GWAS p value file
pval_file='plink.assoc'
snp2p={}
with open(pval_file,'r') as fr:
    i=0
    for line in fr:
        if i==0:
            i+=1
            continue
        a=line.split()
        snp2p[a[1]]=float(a[-2])
# snp2p
'''
# read gene to snp file

gene2snp={}
# corrdinates of snps should be sorted
with open('/scratch/prj/bcn_ml_als/DiseaseCapsule/GWAS_2016/p_0.05/%s/chr' % gwas + str(chrom) +".variant_function.uniqGene",'r') as fr:
    for line in fr:
        chromosome,snp,gene=line.split()[:3]
        if gene in gene2snp:
            gene2snp[gene].append(snp)
        else:
            gene2snp[gene]=[snp]


len(gene2snp)

# gene2snp
'''
topk=128 #top k SNPs sorted by GWAS P value

del_genes=[] # delete the gene if the min GWAS P value of snps in this gene >0.05
for gene in gene2snp.keys():
    if np.min([snp2p[snp] for snp in gene2snp[gene]]) > 0.05:
        del_genes.append(gene)
        continue

    if len(gene2snp[gene])>topk:
        tmp_snp2p={}
        for snp in gene2snp[gene]:
            tmp_snp2p[snp]=snp2p[snp]
        snps=sorted(tmp_snp2p.keys(),key=lambda x :tmp_snp2p[x],reverse=False)[:topk]

        if gene2strand[gene] == '+':
            gene2snp[gene]=sorted(snps,key=lambda x :int(x.split(':')[-1]),reverse=False)
        elif gene2strand[gene] == '-':
            gene2snp[gene]=sorted(snps,key=lambda x :int(x.split(':')[-1]),reverse=True)
        else:
            raise Exception("Cannot find gene:{} in the gene annotation file!".format(gene))


for gene in del_genes:
    del gene2snp[gene]
len(gene2snp)
'''



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


labels_file='/scratch/prj/bcn_ml_als/DiseaseCapsule/GWAS_2016/all_labels.csv'
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

genotype_mapping = {
    0: '0/0',
    1: '0/1',
    -1: './.',
    2: '1/1'
}

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
        return imputed_data #majority_allele
    else:
        return snp_data #-1 #
    
def count_snps_missingness_less_than_50(genotype_data_array, threshold=0.5):
    missing_counts = np.sum(genotype_data_array == -1, axis=0)  # Count occurrences of -1 (missing genotypes)
    selected_snps = missing_counts <= threshold * len(genotype_data_array)
    return np.sum(selected_snps)

def load_majority_alleles_file(filename):
    alleles = {}
    with open(filename, "r") as f:
        for line in f:
            snp, allele = line.strip().split("\t")
            alleles[snp] = int(allele)
    return alleles


for gene in gene2snp.keys(): # 'ACR','CCT8L2'
    #     gene='LINC00308'
    print('processing gene: {}'.format(gene))
    majority_alleles_filename = "/scratch/prj/bcn_ml_als/DiseaseCapsule/GWAS_2016/p_0.05/%s/chr" % gwas + str(chrom) + "/" + gene + "_majority_alleles.txt"
    majority_alleles = load_majority_alleles_file(majority_alleles_filename)
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
    try:
        dataset_X = dataset_X.reshape((len(dataset_X), dataset_X.shape[1]))
        total_snps += dataset_X.shape[1]
        # Impute missing data based on the majority allele from the txt file
        imputed_genotype_array = []
        for i, snp_data in enumerate(dataset_X.T):
            majority_allele = majority_alleles.get(gene2snp[gene][i], None)
            if majority_allele != -1:
                imputed_data = [majority_allele if geno == -1 else geno for geno in snp_data]
                #imputed_data = [geno for geno in imputed_data if geno != -1]
                imputed_genotype_array.append(imputed_data)
        dataset_X = np.array(imputed_genotype_array).T
        dataset_Y = np.array(pheno_new)
        dataset_Y = dataset_Y.reshape((len(dataset_Y), dataset_Y.shape[1]))
        print(dataset_X.shape)
        print(dataset_Y.shape)
    except:
        pass
    if dataset_X.size > 0:
        outfile="chr" +  str(chrom) + "/genes/"+gene+".pkl"
        fw = open(outfile,'wb')
        # pickle.dump(gene,'wb')
        pickle.dump((dataset_X,dataset_Y),fw)
        fw.close()
    
        # gene_pca
        genefile="chr" +  str(chrom) + "/genes/"+gene+".pkl"
        pkl_file=open(genefile,'rb')
        dataset_X,dataset_Y=pickle.load(pkl_file)
        dataset_X.shape
        # dup_rate=round(1-1.0*pd.DataFrame(dataset_X).drop_duplicates().shape[0]/dataset_X.shape[0],3)
        N_snps=dataset_X.shape[1]
        if N_snps > 20:
            encoding_dim = 8
        elif N_snps <= 20 and N_snps > 4:
            encoding_dim = 4
        else:
            encoding_dim = 1
        #umap
        n_comp=encoding_dim
        pca = PCA(n_components=n_comp)
        embedding = pca.fit_transform(dataset_X)
        embedding.shape
        print(pca.explained_variance_ratio_)
        sum(pca.explained_variance_ratio_)
        # -------- markdown --------
        # ## save result
        #save top embedding
        encoded_file="chr" +  str(chrom) + "/pca/"+gene+".pkl"
        with open(encoded_file,'wb') as fw:
            pickle.dump(embedding,fw)
        # pkl_file=open(encoded_file,'rb')
        # dataset_X=pickle.load(pkl_file)


