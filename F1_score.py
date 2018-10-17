#!/usr/bin/python

import sys, os, numpy
from os.path import isfile, join, exists
from sklearn.metrics import f1_score

dir_path = sys.argv[1]
out_file = sys.argv[2]
#out_file = 'F1_score.txt'

if exists(join(dir_path, out_file)):
   os.remove(join(dir_path, out_file))

f_out = open(join(dir_path, out_file), 'w')
for data in os.listdir(dir_path):
    if isfile(join(dir_path, data)):
       continue
    f1_scores = []
    scores = ['score_0fold.txt', 'score_1fold.txt', 'score_2fold.txt']
    for score in scores:
        with open(join(dir_path, data, score)) as f:
            lines = f.readlines()
        pred = []; real = []    
        for line in lines:
            line_split = line.strip().split()
            pred.append(float(line_split[0]))
            real.append(float(line_split[1]))
        pred = [int(i >= 0.5) for i in pred]
        real = [int(i) for i in real]
        f1_scores.append(f1_score(real, pred))
    print >> f_out, "{} {:.5f}".format(data, numpy.mean(f1_scores))

f_out.close()
