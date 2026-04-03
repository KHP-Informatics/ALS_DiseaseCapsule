import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, confusion_matrix

import json
import csv
import numpy as np
import os
import sys
import time
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import pickle
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


import torch
import torch.nn.functional as F
from torch.nn.functional import relu,tanh
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#from livelossplot import PlotLosses
#from torchsummary import summary

from tensorflow.keras.utils import plot_model


test_set_name = sys.argv[1]

#set randome seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) #fix hash seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 1521024
seed_torch(seed)


class ConvCaps2D(nn.Module):
    def __init__(self):
        super(ConvCaps2D, self).__init__()
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels = primary_capslen,
                                                 kernel_size=(1,ks), stride=stride) for _ in range(filters)])
    def squash(self, tensor, dim=-1):
        norm = (tensor**2).sum(dim=dim, keepdim = True) # norm.size() is (None, 1152, 1)
        scale = norm / (1 + norm) # scale.size()  is (None, 1152, 1)
        return scale*tensor / torch.sqrt(norm)
    def forward(self, x):
        outputs = [capsule(x).view(x.size(0), primary_capslen, -1) for capsule in self.capsules] # 32 list of (None, 1, 8, 36)
        outputs = torch.cat(outputs, dim = 2).permute(0, 2, 1)  # outputs.size() is (None, 1152, 8)
        return self.squash(outputs)


class Caps1D(nn.Module):
    def __init__(self):
        super(Caps1D, self).__init__()
        self.num_iterations = num_iterations
        self.num_caps = 2 # equals to class number
        self.num_routes= (int((neurons-ks)/stride)+1)*filters
        self.in_channels=primary_capslen
        self.out_channels=digital_capslen
        self.W = nn.Parameter(torch.randn(self.num_caps,self.num_routes, self.in_channels, self.out_channels)) # class,weight,len_capsule,capsule_layer
#         self.W = nn.Parameter(torch.randn(3, 3136, 8, 32)) # num_caps, num_routes, in_channels, out_channels
    def softmax(self, x, dim = 1):
        transposed_input = x.transpose(dim, len(x.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
        #softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(x.size()) - 1)
    def squash(self, tensor, dim=-1):
        norm = (tensor**2).sum(dim=dim, keepdim = True) # norm.size() is (None, 1152, 1)
        scale = norm / (1 + norm)
        return scale*tensor / torch.sqrt(norm)
    # Routing algorithm
    def forward(self, u):
        # u.size() is (None, 1152, 8)
        '''
        From documentation
        For example, if tensor1 is a j x 1 x n x m Tensor and tensor2 is a k x m x p Tensor,
        out will be an j x k x n x p Tensor.
        We need j = None, 1, n = 1152, k = 10, m = 8, p = 16
        '''
        u_ji = torch.matmul(u[:, None, :, None, :], self.W) # u_ji.size() is (None, 10, 1152, 1, 16)
        b = Variable(torch.zeros(u_ji.size())) # b.size() is (None, 10, 1152, 1, 16)
        b = b.to(device) # using gpu
        for i in range(self.num_iterations):
            c = self.softmax(b, dim=2)
            v = self.squash((c * u_ji).sum(dim=2, keepdim=True)) # v.size() is (None, 10, 1, 1, 16)
            if i != self.num_iterations - 1:
                delta_b = (u_ji * v).sum(dim=-1, keepdim=True)
                b = b + delta_b
        # Now we simply compute the length of the vectors and take the softmax to get probability.
        v = v.squeeze()
        classes = (v ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)
        #classes = F.softmax(classes, dim=1)
        return classes


class CapsNet(nn.Module):
    def __init__(self):
#         super().__init__() #py3
        super(CapsNet, self).__init__() #py2
        self.fc1 = nn.Linear(top_k,neurons)
        self.dropout1 = nn.Dropout(p=dropout)
        self.primaryCaps = ConvCaps2D()
        self.digitCaps = Caps1D()
    def forward(self, x):
        x = act(self.dropout1(self.fc1(x)))
        x = self.primaryCaps(x)
        x = self.digitCaps(x)
        return x


# train on cuda if available
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def train_model(model, criterion, optimizer, num_epochs=30):
    #liveloss = PlotLosses()
    model = model.double()
    model = model.to(device)
    for epoch in range(num_epochs):
        logs = {}
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.double()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            prefix = ''
            if phase == 'validation':
                prefix = 'val_'
            logs[prefix + 'log loss'] = epoch_loss.item()
            logs[prefix + 'accuracy'] = epoch_acc.item()
            train_loss_values.append(epoch_loss.item())  # Store training loss
            validation_loss_values.append(epoch_loss.item())  # Store validation loss
        scheduler.step()
        # liveloss.update(logs)
#         liveloss.draw()

#use the best param, id:89
neurons=150
dropout=0.5
primary_capslen=4
digital_capslen=16
ks=5
stride=2
filters=32
num_iterations=3 #danymic routing iterations

##
initial_lr=0.0001
batch_size=128
epochs=30
act=relu

# feature_file='sigSNPs_pca_ind.features.pkl'
## predict test dataset
feature_file = '/scratch/prj/bcn_ml_als/DiseaseCapsule/GWAS_2016/p_0.05/meta/' + 'sigSNPs_pca.features.pkl'
dataset_X,_=pd.read_pickle(open(feature_file,'rb'))
dataset_X=np.array(dataset_X)
# _,dataset_Y=pickle.load(open('/scratch/prj/bcn_ml_als/DiseaseCapsule/GWAS_2016/p_0.05/lmm/chr1/genes/A3GALT2.pkl','rb'))
_,dataset_Y=pickle.load(open('/scratch/prj/bcn_ml_als/DiseaseCapsule/GWAS_2016/p_0.05/meta/chr1/genes/A3GALT2.pkl','rb'))
test_idx = [int(line.strip()) for line in open(f"/scratch/prj/bcn_ml_als/DiseaseCapsule/GWAS_2016/p_0.05/meta/{test_set_name}/test.idx", 'r')]
y_test = dataset_Y[test_idx]

#cohort = sys.argv[1]

# test dataset
#te_idx = [int(line.strip()) for line in open(f"/scratch/prj/bcn_ml_als/DiseaseCapsule/GWAS_2016/p_0.05/meta/{test_set_name}/test.idx", 'r')]
x_test = dataset_X[test_idx]
#y_test = dataset_Y[te_idx]
y_test = np.argmax(y_test, axis=1)
#x_test=np.array(x_test)
#x_test = x_test.double()
top_k=x_test.shape[1]
x_test = x_test.reshape(x_test.shape[0], 1, 1, top_k)
#x_test = torch.tensor(x_test, dtype=torch.double).to(device)
in_test = Variable(torch.tensor(x_test).to(device))
model = CapsNet()
model = model.double()
model.load_state_dict(torch.load(f'/scratch/prj/bcn_ml_als/DiseaseCapsule/GWAS_2016/p_0.05/meta/{test_set_name}/capsule_pca_1.0model_1.pt'))  # Replace 'model_filename.pt' with the actual filename

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Make predictions on the test data
model.eval()
with torch.no_grad():
    y_pred1 = model(in_test.double()).detach().cpu().numpy()

# Convert the model predictions to class labels using argmax
y_pred = np.argmax(y_pred1, axis=1)
y_true = y_test

auc = roc_auc_score(y_true, y_pred1[:, 1], multi_class='ovr')
# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred1[:, 1])
plt.plot(fpr, tpr) #, label='AUC = {:.3f}'.format(auc))
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line indicating random classification
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig(f'{test_set_name}/capsule_AUC.png')

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
acc = round((tp + tn) * 1. / (tp + fp + tn + fn),3)
precision = round(tp*1./(tp+fp),3)
recall = round(tp*1./(tp+fn),3)
f1=round(2*(precision*recall)/(precision+recall),3)

print('all done...')
print('Accuracy:',acc)
print("Pression: ", precision)
print("Recall:", recall)
print("F1: ",f1)
print("AUC: ",auc)
print("TP={}, TN={}, FP={}, FN={}".format(tp,tn,fp,fn))
#print(te_idx)
print(y_pred)

with open(f'/scratch/prj/bcn_ml_als/DiseaseCapsule/GWAS_2016/p_0.05/meta/capsule.out.csv','a') as fw:
    fw.write(','.join(['capsule_0.05']+list(map(str,[test_set_name,acc,precision,recall,f1,auc,tp,tn,fp,fn])))+'\n')



