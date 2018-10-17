#!/usr/bin/bash
#!/usr/bin/bash

Threadnum=3
tmp_file="/tmp/$$.fifo"
mkfifo $tmp_file
exec 6<> $tmp_file
rm $tmp_file
for((i=0; i<$Threadnum; ++i))
do
    echo ""
done >&6

# In the training and testing phase
for eachTF in `ls ./rawdata/`
do
    read -u6
    {
	echo $eachTF
	python encoding.py ./rawdata/${eachTF}/positive.fasta ./rawdata/${eachTF}/negative.fasta ./rawdata/${eachTF}/data/datalabel.hdf5 -m ./mappers/3mer.txt -c 120 -s 10 --no-reverse -kmer 3 -run 'ws'
	
	echo "" >&6
    }&
done
wait
exec 6>&-
