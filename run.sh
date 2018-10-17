#!/usr/bin/bash


# In the training and testing phase
for eachTF in `ls ./rawdata/`
do 
	echo $eachTF
	if [ -d ./model/$eachTF ]; then
	   echo $eachTF 'has existed.'
	   continue
	fi
	python train_val_test.py -datalable ./rawdata/$eachTF/data/datalabel.hdf5 -k 3 -run 'ws' -batchsize 300 -params 12

done

