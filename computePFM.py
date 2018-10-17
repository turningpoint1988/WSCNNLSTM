#!/usr/bin/env python

import sys,os,numpy as np,h5py
from os.path import join,exists
from os import makedirs,system
from models import WSCNNwithNoisy
from keras.models import load_model
from keras import backend as K
from keras.optimizers import Adadelta
from keras.models import Model
   
    
def main():
    root='/home/zqh/Keras/CNN-LSTM'
    kernelnum = 16
    kernelsize = 24
    trim_flag = False
  
    model_dir = join(root, 'model')
    allnames = os.listdir(model_dir)
    for name in allnames:
        if exists(join(root, 'PFM', name)):
           print "the file {} is existed. continuing...".format(name)
           continue
        datalabel = join(root, 'rawdata', name, 'data/datalabel.hdf5')
        model_weights = join(model_dir, name, 'bestmodel.hdf5')
        with h5py.File(datalabel, 'r') as f:
           X_test = np.asarray(f['data'])
           y_test = np.asarray(f['label'])
        seqs_num = X_test.shape[0]; instance_num = X_test.shape[1]
        instance_len = X_test.shape[2]; instance_dim = X_test.shape[3]
        print 'there are %d seqences, each of which is a %d*%d*%d array' %(seqs_num, instance_num, instance_len, instance_dim)
        input_shape = (instance_num, instance_len, instance_dim)
        params = {'DROPOUT': 0.5, 'DELTA': 1e-06, 'MOMENT': 0.99}
        model = WSCNNwithNoisy(input_shape, params)
        model.load_weights(model_weights)
        intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[0].output)
        intermediate_output = intermediate_layer_model.predict(X_test)
        print intermediate_output.shape
        #print intermediate_output[0]

        activator = [0. for x in range(kernelnum)]
        for seq_index in range(seqs_num):
            seq = X_test[seq_index]
            score = intermediate_output[seq_index]
            for kernel in range(kernelnum):
                score_by_kernel = score[:,:,kernel]
                if np.max(score_by_kernel) > 0:
                    row, col = score_by_kernel.shape
                    maxposi = np.argmax(score_by_kernel)
                    m, n = divmod(int(maxposi), col)
                    data_s = (m,n)
                    data_e = (m,n+kernelsize)
                    data_ans = seq[m,data_s[1]:data_e[1],:]
                    assert len(data_ans) == kernelsize, "Wrong!"
                    activator[kernel] += data_ans

        pfm = compute_pfm(activator, kernelnum, kernelsize, trim_flag)
        writeFile(join(root, 'PFM'),name, pfm)

def compute_pfm(activator,kernelnum,kernelsize,trim_flag):
    pfm = []
    information = np.zeros((kernelnum,kernelsize))
    for num in range(kernelnum):
        try: temp = activator[num].astype(np.float32)
        except: print activator[num]; continue
        for i in range(temp.shape[0]):
            sum_ = np.sum(temp[i])
            temp[i] = temp[i] / sum_
            information[num, i] = 2 + np.sum(temp[i] * np.log2(temp[i]+1e-5))
        if trim_flag:
           pfm_s, pfm_e = trim_pfm(information[num])
           temp = temp[pfm_s:pfm_e, :]
        pfm.append(temp)
    return pfm

def trim_pfm(info):        
    max_ic = np.max(info)
    ic_threshold = np.max([0.1*max_ic, 0.1])
    w = info.shape[0]
    pfm_st = 0
    pfm_sp = w
    for i in range(w):
        if info[i] < ic_threshold:
            pfm_st += 1
        else:
            break
    for inf in reversed(info):    
        if inf < ic_threshold:            
            pfm_sp -= 1            
        else:
            break
    return pfm_st, pfm_sp

def writeFile(out_dir,name,pfm):
    if not exists(join(out_dir, name)):
       print "creating %s" % join(out_dir, name)
       os.mkdir(join(out_dir, name))
    out_f = open(join(out_dir, name, 'pfm.txt'), 'w')
    out_f.write("MEME version 5.0.0\n\n")
    out_f.write("ALPHABET= ACGT\n\n")
    out_f.write("strands: +\n\n")
    out_f.write("Background letter frequencies\n")
    out_f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
    for i in range(len(pfm)):
        out_f.write("MOTIF " + name + "%s\n" % str(i+1))
        out_f.write("letter-probability matrix: alength= 4 w= 24 nsites= 19 E= 0\n")
        current_pfm = pfm[i]
        for row in range(current_pfm.shape[0]):
            for col in range(current_pfm.shape[1]):
                out_f.write("%g " % current_pfm[row,col])
            out_f.write("\n")
        out_f.write("\n")
    out_f.close()

if __name__ == "__main__": main()
