#/usr/bin/python

import sys, os
from os.path import exists, isfile

dir_path = sys.argv[1]
if exists(dir_path + '/ROCAUC.txt'):
   os.remove(dir_path + '/ROCAUC.txt')
if exists(dir_path + '/PRAUC.txt'):
   os.remove(dir_path + '/PRAUC.txt')

file_names = os.listdir(dir_path)
f_roc = open(dir_path + '/ROCAUC.txt', 'w')
f_pr = open(dir_path + '/PRAUC.txt', 'w')
for name in file_names:
    if isfile(dir_path + '/%s' %name): continue
    with open(dir_path + '/%s/metrics.txt' % name, 'r') as f:
        lines = f.readlines()
        line = lines[-1].strip().split()
        print >> f_roc, "{} {}".format(name, line[0])
        print >> f_pr, "{} {}".format(name, line[1])

f_roc.close(); f_pr.close()

