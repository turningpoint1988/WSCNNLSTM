#!/usr/bin/env python
# encoding: utf-8

import numpy as np, sys, math, os, h5py
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from multiprocessing import Pool
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Generate ids for k-flods cross-validation
def Id_k_folds(seqs_num, k_folds, ratio):
    train_ids = []; test_ids = []; valid_ids = []
    if k_folds == 1:
       train_num = int(seqs_num*0.7)
       test_num = seqs_num - train_num
       valid_num = int(train_num*ratio)
       train_num = train_num - valid_num
       index = range(seqs_num)
       train_ids.append(np.asarray(index[:train_num]))
       valid_ids.append(np.asarray(index[train_num:train_num+valid_num]))
       test_ids.append(np.asarray(index[train_num+valid_num:]))
    else:
       each_fold_num = int(math.ceil(seqs_num/k_folds))
       for fold in range(k_folds):
           index = range(seqs_num)
           index_slice = index[fold*each_fold_num:(fold+1)*each_fold_num]
           index_left = list(set(index) - set(index_slice))
           test_ids.append(np.asarray(index_slice))
           train_num = len(index_left) - int(len(index_left) * ratio)
           train_ids.append(np.asarray(index_left[:train_num]))
           valid_ids.append(np.asarray(index_left[train_num:]))
       
    return (train_ids, test_ids, valid_ids)

# Compute the roc AUC and the precision-recall AUC
def ComputeAUC(y_pred, y_real):
    # roc_auc_score(y_real, y_pred)
    fpr, tpr, thresholds = roc_curve(y_real, y_pred)
    roc_auc = auc(fpr, tpr)
    # average_precision_score(y_real, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_real,y_pred)
    pr_auc = auc(recall,precision)    
    
    return (roc_auc, pr_auc)

# Compute f1 score
def ComputeF1(y_pred, y_real):

    score = f1_score(y_real, y_pred)   
    
    return (score)     

#Compute the pearson corelation coefficient
def ComputePCC(y_pred, y_real):
    # pearson coefficient
    coeff1, pvalue = stats.pearsonr(y_pred, y_real)
    coeff2, pvalue = stats.spearmanr(y_pred, y_real)
    
    return (coeff1, coeff2)

# Generate random hyper-paramter settings
def RandomSample():
    space = {
    'DROPOUT': hp.choice( 'drop', (0.2, 0.5)),
    'DELTA': hp.choice( 'delta', (1e-06, 1e-08)),
    'MOMENT': hp.choice( 'moment', (0.9, 0.99, 0.999 ))
    }
    params = sample(space)
    return params

# Generate all hyper-paramter settings
def AllSample():
    DROPOUT = [0.2, 0.5]
    DELTA = [1e-06, 1e-08]
    MOMENT = [0.9, 0.99, 0.999]
    space = []
    for drop in DROPOUT:
        for delta in DELTA:
            for moment in MOMENT:
                space.append({'DROPOUT': drop, 'DELTA': delta, 'MOMENT': moment})
    return space

# select the best paramter setting
def SelectBest(history_all, file_path, fold, monitor='val_loss'):
    if monitor == 'val_loss':
       loss = 100000.
       for num, History in history_all.items():
           if np.min(History.history['val_loss']) < loss:
              best_num = int(num)
              loss = np.min(History.history['val_loss'])
    else:
       acc = 0.
       for num, History in history_all.items():
           if np.max(History.history['val_acc']) > acc:
              best_num = int(num)
              acc = np.max(History.history['val_acc'])
    
    del_num = range(len(history_all))
    del_num.pop(best_num)
    # delete the useless model paramters
    for num in del_num:
       os.remove(file_path + 'params%d_bestmodel_%dfold.hdf5' %(num, fold))
    
    return best_num

# plot and save the training process
def PlotandSave(History, filepath, fold, monitor='val_loss'):
    if monitor == 'val_loss':
       train_loss = History.history['loss']
       valid_loss = History.history['val_loss']
       x = range(len(train_loss))
    
       plt.figure(num = fold)
       plt.title('mode loss')
       plt.ylabel('loss')
       plt.xlabel('epoch')
       plt.plot(x, train_loss, 'r-', x, valid_loss, 'g-')
       plt.legend(['train_loss', 'valid_loss'], loc = 'upper left')
       #plt.show()
    else:   
       train_acc = History.history['acc']
       valid_acc = History.history['val_acc']
       x = range(len(train_acc))
    
       plt.figure(num = fold)
       plt.title('model accuracy')
       plt.ylabel('accuracy')
       plt.xlabel('epoch')
       plt.plot(x, train_acc, 'r-', x, valid_acc, 'g-')
       plt.legend(['train_acc', 'valid_acc'], loc = 'upper left')
       #plt.show()
    
    plt.savefig(filepath, format = 'png')

    
    
