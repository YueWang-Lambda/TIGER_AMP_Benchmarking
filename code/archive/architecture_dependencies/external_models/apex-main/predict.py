import os
import json
#from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
from torch.autograd import Variable
from scipy import stats
import pandas as pd 
from sklearn.model_selection import KFold
import pickle
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import os.path
from Bio import SeqIO
import string
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from AMP_DL_model_twohead import AMP_model
#from propy.AAComposition import CalculateAADipeptideComposition
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import stats
from utils import *
from scipy import sparse
import sys
from optparse import OptionParser
import copy
import pandas as pd


col = ['E. coli ATCC11775', 'P. aeruginosa PAO1', 'P. aeruginosa PA14', 'S. aureus ATCC12600', 'E. coli AIG221', 'E. coli AIG222', 'K. pneumoniae ATCC13883', 'A. baumannii ATCC19606', 'A. muciniphila ATCC BAA-835', 'B. fragilis ATCC25285', 'B. vulgatus ATCC8482', 'C. aerofaciens ATCC25986', 'C. scindens ATCC35704', 'B. thetaiotaomicron ATCC29148', 'B. thetaiotaomicron Complemmented', 'B. thetaiotaomicron Mutant', 'B. uniformis ATCC8492', 'B. eggerthi ATCC27754', 'C. spiroforme ATCC29900', 'P. distasonis ATCC8503', 'P. copri DSMZ18205', 'B. ovatus ATCC8483', 'E. rectale ATCC33656', 'C. symbiosum', 'R. obeum', 'R. torques', 'S. aureus (ATCC BAA-1556) - MRSA', 'vancomycin-resistant E. faecalis ATCC700802', 'vancomycin-resistant E. faecium ATCC700221', 'E. coli Nissle', 'Salmonella enterica ATCC 9150 (BEIRES NR-515)', 'Salmonella enterica (BEIRES NR-170)', 'Salmonella enterica ATCC 9150 (BEIRES NR-174)', 'L. monocytogenes ATCC 19111 (BEIRES NR-106)']

max_len = 52 # maximun peptide length

word2idx, idx2word = make_vocab()
emb, AAindex_dict = AAindex('./aaindex1.csv', word2idx)
vocab_size = len(word2idx)
emb_size = np.shape(emb)[1]


model_num = 8
repeat_num = 5


f = open('./best_key_list', 'r')
lines = f.readlines()
f.close()

model_list = []
for line in lines:
  parsed = line.strip('\n').strip('\r')
  model_list.append(parsed)


all_list = []
ensemble_num = model_num * repeat_num

deep_model_list = []
for a_model_name in model_list:
  for a_en in range(repeat_num):
    key = 'trained_all_model_'+a_model_name+'_ensemble_'+str(a_en)

    model = torch.load('./trained_models/'+key)
    model.eval()
    deep_model_list.append(model)






seq_list = []
f = open('./test_seqs.txt', 'r')
lines = f.readlines()
f.close()

for line in lines:
	seq_list.append(line.strip('\n').strip('\r'))

seq_list = np.array(seq_list)

ensemble_counter = 0
for ensemble_id in range(ensemble_num):

	AMP_model = deep_model_list[ensemble_id].cuda().eval()

	data_len = len(seq_list)
	batch_size = 3000 #change according to your GPU memory
	for i in range(int(math.ceil(data_len/float(batch_size)))):
		if (i*batch_size) % 1000 == 0:
			print ('progress', i*batch_size, data_len)

		seq_batch = seq_list[i*batch_size:(i+1)*batch_size]
		seq_rep, _, _ = onehot_encoding(seq_batch, max_len, word2idx)

		X_seq = torch.LongTensor(seq_rep).cuda()


		AMP_pred_batch = AMP_model(X_seq).cpu().detach().numpy()
		AMP_pred_batch = 10**(6-AMP_pred_batch) #transform back to MICs

		if i == 0:
			AMP_pred = AMP_pred_batch
		else:
			AMP_pred = np.vstack([AMP_pred, AMP_pred_batch])

	if ensemble_id == 0:
		AMP_sum = AMP_pred
	else:
		AMP_sum += AMP_pred
	ensemble_counter += 1

AMP_pred = AMP_sum / float(ensemble_counter)

df = pd.DataFrame(data=AMP_pred, columns=col, index=seq_list)
print (df)

df.to_csv('Predicted_MICs.csv')
