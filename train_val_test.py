#!/usr/bin/env python
# encoding: utf-8

import numpy as np, argparse, sys, h5py
from os.path import join, abspath, dirname, exists
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta
from keras.models import load_model
from models import DeepBind, DanQ, WSCNNwithNoisy, WSCNNwithMax, WSCNNwithAve, WSCNNLSTMwithNoisy, WSCNNLSTMwithMax, WSCNNLSTMwithAve
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Convert sequence and target")
    parser.add_argument('-datalable', dest='datalable', type=str, help='Positive data for training, testing')
    parser.add_argument('-k', dest='k_folds', type=int, default=3, help='k-folds cross-validation')
    parser.add_argument('-kernelsize', dest='kernelsize', type=int, default=24, help='the kernelsize of convolutional layer')
    parser.add_argument('-batchsize', dest='batchsize', type=int, default=300, help='the size of one batch')
    parser.add_argument('-ratio', dest='ratio', type=float, default=0.125, help='the propotion of validation data over the training data')
    parser.add_argument('-params', dest='params', type=int, default=1, help='the number of paramter settings')
    parser.add_argument('-train', dest='train', action='store_true', default=True, help='only test step')
    parser.add_argument('-plot', dest='plot', action='store_true', default=True, help='only test step')
    parser.add_argument('-run', dest='run', type=str, default='nows', help='three encoding methods, including ws, nows')
    
    return parser.parse_args()

def main():

    file_path = dirname(abspath(__file__))
    args = parse_args()
    
    # print the current dataset name
    name = (args.datalable).split('/')[-3]
    print 'working on %s now' % name
    
    # convert raw data 
    if args.run == 'nows':
       print 'using no-ws encoding method.'
       with h5py.File(args.datalable, 'r') as f:
           seqs_vector = np.asarray(f['data'])
           labels = np.asarray(f['label'])
       seqs_num = seqs_vector.shape[0]; seqs_len = seqs_vector.shape[1]
       seqs_dim = seqs_vector.shape[2]
       print 'there are %d seqences, each of which is a %d*%d array' %(seqs_num, seqs_len, seqs_dim)
       input_shape = (seqs_len, seqs_dim)
    elif args.run == 'ws':
       print 'using ws encoding method.'
       with h5py.File(args.datalable, 'r') as f:
           seqs_vector = np.asarray(f['data'])
           labels = np.asarray(f['label'])
       seqs_num = seqs_vector.shape[0]; instance_num = seqs_vector.shape[1]
       instance_len = seqs_vector.shape[2]; instance_dim = seqs_vector.shape[3]
       print 'there are %d seqences, each of which is a %d*%d*%d array' %(seqs_num, instance_num, instance_len, instance_dim)
       input_shape = (instance_num, instance_len, instance_dim)
    else:
       print >> sys.stderr, 'invalid command!';sys.exit(1)
    
    # k-folds cross-validation
    indices = np.arange(seqs_num)
    np.random.shuffle(indices)
    seqs_vector = seqs_vector[indices]
    labels = labels[indices]

    train_ids, test_ids, valid_ids = Id_k_folds(seqs_num, args.k_folds, args.ratio)
    rocauc = []; prauc = []
    
    space = AllSample()
    if not exists(file_path + '/model/%s' % name):
       print 'Building ' + file_path + '/model/%s' % name
       os.mkdir(file_path + '/model/%s' % name)
    f_params = open(file_path + '/model/%s/params.txt' % name, 'w')
    for fold in range(args.k_folds):
        X_train = seqs_vector[train_ids[fold]]
        y_train = labels[train_ids[fold]]
        X_test = seqs_vector[test_ids[fold]]
        y_test = labels[test_ids[fold]]
        X_valid = seqs_vector[valid_ids[fold]]
        y_valid = labels[valid_ids[fold]]
       
        if args.train:
           history_all = {}
           for params_num in range(args.params):
               params = space[params_num]
               print "the {}-th paramter setting of the {}-th is {}".format(params_num, fold, params)
               print >> f_params, "the {}-th paramter setting of the {}-th is {}".format(params_num, fold, params)
               print 'Building model...'
               if args.run == 'nows':
                  model = DeepBind(input_shape, params)
               else:
                  model = WSCNNLSTMwithNoisy(input_shape, params)
               checkpointer = ModelCheckpoint(filepath=file_path + '/model/%s/params%d_bestmodel_%dfold.hdf5' 
                                                % (name, params_num, fold), monitor='val_acc', verbose=1, save_best_only=True)
               earlystopper = EarlyStopping(monitor='val_acc', patience=6, verbose=1)

               print 'Training model...'
               myoptimizer = Adadelta(epsilon=params['DELTA'], rho=params['MOMENT'])
               model.compile(loss='binary_crossentropy', optimizer=myoptimizer, metrics=['accuracy'])
               History = model.fit(X_train, y_train, epochs=60, batch_size=args.batchsize, shuffle=True,
                                   validation_data=(X_valid, y_valid), callbacks=[checkpointer, earlystopper], verbose=2)
               history_all[str(params_num)] = History
           best_num = SelectBest(history_all, file_path + '/model/%s/' % name, fold, 'val_acc')
           if args.plot: PlotandSave(history_all[str(best_num)], file_path + '/model/%s/figure_%dfold.png' % (name, fold), fold, 'val_acc')
          
        print 'Testing model...'
        # load_model('')
        model.load_weights(file_path + '/model/%s/params%d_bestmodel_%dfold.hdf5' % (name, best_num, fold))
        results = model.evaluate(X_test, y_test)
        print results
        y_pred = model.predict(X_test, batch_size=args.batchsize, verbose=1)
        y_pred = np.asarray([y[0] for y in y_pred])
        y_real = np.asarray([y[0] for y in y_test])
        with open(file_path + '/model/%s/score_%dfold.txt' % (name, fold), 'w') as f:
           assert len(y_pred) == len(y_real), 'dismathed!'
           for i in range(len(y_pred)):
               print >> f, '{:.5f} {}'.format(y_pred[i], y_real[i])

        print 'Calculating AUC...'
        auroc, auprc = ComputeAUC(y_pred, y_real)
        rocauc.append(auroc); prauc.append(auprc)
       
    f_params.close()
    print "the average rocauc is {} and the average prauc is {}".format(np.mean(rocauc), np.mean(prauc))
    outfile = file_path + '/model/%s/metrics.txt' % (name)
    with open(outfile,'w') as f:
        for i in range(len(rocauc)):
            print >> f, "{:.5f} {:.5f}".format(rocauc[i], prauc[i])
        print >> f, "{:.5f} {:.5f}".format(np.mean(rocauc), np.mean(prauc))
    
if __name__ == '__main__': main()    
    
    
    
    
    
    
