#!/usr/bin/env python

# -------- markdown --------
# - Should run under python3 in order to keep the seed not changed.


import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import sys

file = sys.argv[1]
random_state = int(sys.argv[2])

# -------- markdown --------
# # use all samples. 10% for test

# file=

def indx(lab):
    #     lab = np.argmax(lab,axis=1)
    p = []  # positive samples index-- ALS
    n = []  # negative samples index-- Non-ALS
    for i in range(len(lab)):
        if lab[i] == 0:
            p.append(i)
        else:
            n.append(i)
    return p, n
'''
def dataset(X, Y, test_ratio):
    lab = np.argmax(Y, axis=1)
    pos_s, neg_s = indx(lab)
    N = len(lab)
    idx = range(N)
    N_te = int(N * test_ratio) / 5 * 5  # number of test samples
    N_tr = N - N_te  # number of training samples
    pos_s_te = int(N_te * 0.5)
    neg_s_te = int(N_te * 0.5)
    random.shuffle(pos_s)
    random.shuffle(neg_s)
    pos_idx_te = pos_s[:pos_s_te]
    neg_idx_te = neg_s[:neg_s_te]
    te_idx = pos_idx_te + neg_idx_te
    tr_idx = list(set(idx) - set(te_idx))
    random.shuffle(te_idx)
    random.shuffle(tr_idx)
    tr_X = X[tr_idx]
    tr_Y = Y[tr_idx]
    te_X = X[te_idx]
    te_Y = Y[te_idx]
    with open('test.idx','w') as fw:
        fw.write('\n'.join(list(map(str,te_idx))))
    return tr_X, tr_Y, te_X, te_Y
'''

#

#labels_file = 'labels.csv'
labels_file = '../all_filled/labels.csv'
labels_df = pd.read_csv(labels_file, index_col=0, sep='\t')
#labels_file = 'labels_sIB.csv'
#labels_df = pd.read_csv(labels_file, index_col=0, sep='\t')

# Separate cases and controls based on their labels
cases_df = labels_df[labels_df['Pheno'] == 2]  # Assuming label 2 represents cases
controls_df = labels_df[labels_df['Pheno'] == 1]  # Assuming label 1 represents controls

# Randomly sample 30 cases and 9000 controls
num_cases = 30
num_controls = 9000


cases_sample = cases_df.sample(n=num_cases, random_state=random_state)
controls_sample = controls_df.sample(n=num_controls, random_state=random_state)

# Concatenate the sampled cases and controls
result_df = pd.concat([cases_sample, controls_sample])

# Shuffle the DataFrame to mix cases and controls
result_df = result_df.sample(frac=1, random_state=random_state)

# Save the index values to test.idx
result_df.index.to_numpy().tofile("../all_filled_%s/test.idx" % str(file), sep='\n')

idx_file = "../all_filled_%s/test.idx" % file
with open(idx_file, "r") as f:
    test_set_indices = [int(line.strip()) for line in f]

ids_csv = labels_df.FID.tolist()

labels_df['vars'] = None

# the output is a vector for probability in DL
lab_num = {1: [1, 0], # negative, max_idx=0
           2: [0, 1]} # positive, max_idx=1


pheno_new = []
for i in labels_df.Pheno.tolist():
    pheno_new.append(lab_num[i])

d = {"Pheno": pheno_new, "Vars":labels_df.vars}
dataset_ = pd.DataFrame(d)


dataset_Y = np.array(dataset_.Pheno.tolist())
dataset_X = dataset_Y

random.seed(123)
#x_train_val, y_train_val, x_test, y_test = dataset(dataset_X,dataset_Y,test_ratio=0.1)

test_idx=[int(i.strip()) for i in open(idx_file,'r')]
train_val_idx=[str(i) for i in range(dataset_Y.shape[0]) if i not in test_idx]

# -------- markdown --------
# # five fold cross validation

idx1_list=[int(idx) for idx in train_val_idx if labels_df.Pheno[int(idx)]==1]
idx2_list=[int(idx) for idx in train_val_idx if labels_df.Pheno[int(idx)]==2]

random.seed(1991)
random.shuffle(idx1_list)
random.shuffle(idx2_list)
idx2_list[:3]

len(idx1_list)
len(idx2_list)


n = min(len(idx1_list), len(idx2_list)) - 2
# change from: n=2670
random.seed(1991)
idx1_listp=random.sample(idx1_list,n)
idx2_listp=random.sample(idx2_list,n)

len(idx1_listp)
len(idx2_listp)

fwt=open('train.idx','w')
fwv=open('validation.idx','w')
seed=123
k=5 #five fold cv
for _ in range(k):
    random.seed(1991)
    val_idx1=random.sample(idx1_listp,int(n/k))
    val_idx2=random.sample(idx2_listp,int(n/k))
    val_idx=val_idx1+val_idx2
    random.seed(1991)
    random.shuffle(val_idx)
    fwv.write(','.join(list(map(str,val_idx)))+'\n')
    for e in val_idx1:idx1_listp.remove(e)
    for e in val_idx2:idx2_listp.remove(e)
    train_idx=[int(i) for i in train_val_idx if int(i) not in val_idx]
    idx1_tmp=[int(idx) for idx in train_idx if labels_df.Pheno[int(idx)]==1]
    idx2_tmp=[int(idx) for idx in train_idx if labels_df.Pheno[int(idx)]==2]
    print('control:{}'.format(len(idx1_tmp)))
    print('case:{}'.format(len(idx2_tmp)))
    seed+=10
    np.random.seed(seed)
    idx2_tmp2=np.random.choice(idx2_tmp,len(idx1_tmp)-len(idx2_tmp),replace=True)
    idx2_tmp=idx2_tmp+list(idx2_tmp2)
    print('case_new:{}'.format(len(idx2_tmp)))
    train_idx=idx1_tmp+idx2_tmp
    print('all:{}'.format(len(train_idx)))
    print('-'*8)
    random.seed(1991)
    random.shuffle(train_idx)
    fwt.write(','.join(list(map(str,train_idx)))+'\n')


fwt.close()
fwv.close()
print('');




# -------- markdown --------
# # train+validation data index, get balanced control and case !


test_idx=[int(i.strip()) for i in open('test.idx','r')]
train_val_idx=[str(i) for i in range(dataset_Y.shape[0]) if i not in test_idx]
len(train_val_idx)
train_val_idx[:10]

random.seed(1991)
len(idx1_list)
len(idx2_list)

idx2_list_bal = random.sample(idx2_list, len(idx1_list)-len(idx2_list))
train_idx2=idx2_list+idx2_list_bal+idx1_list

'''
change from: 
idx2_list_bal = random.sample(idx2_list, len(idx1_list)-len(idx2_list)-len(idx2_list))
train_idx2=idx2_list+idx2_list+idx2_list_bal+idx1_list
'''

fwt=open('train_val.balanced.idx','w')
fwu=open('train_val.unique.idx','w')

random.seed(1991)
random.shuffle(train_idx2)
fwt.write('\n'.join(list(map(str, train_idx2)))+'\n')

train_idx3=list(set(train_idx2))
len(train_idx3)
random.seed(1991)
random.shuffle(train_idx3)
len(train_idx3)

fwu.write('\n'.join(list(map(str, train_idx3)))+'\n')

len(train_idx2)

fwt.close()
fwu.close()
exit()

'''
# script for generating test fam file
def get_sample_ids_from_idx_file(idx_file, fam_file):
    # Read the idx file and extract the indices
    with open(idx_file, 'r') as idx:
        indices = [int(line.strip()) for line in idx]
    # Read the fam file and extract the sample IDs based on the indices
    sample_ids = []
    with open(fam_file, 'r') as fam:
        for i, line in enumerate(fam):
            if i in indices:
                sample_id = line.split()[0]  # Extract the first column (sample ID)
                sample_ids.append(sample_id)
    return sample_ids

idx_file = 'test.idx'
fam_file = '/scratch/users/k20066216/vcf_imputed/%s.fam' % file
sample_ids = get_sample_ids_from_idx_file(idx_file, fam_file)


with open('test.samples.fam', 'w') as fw:
    fw.write('\n'.join(sample_ids))
    
'''
