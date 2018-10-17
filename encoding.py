import argparse,pwd,os,numpy as np,h5py,sys
from os.path import exists,dirname
from os import makedirs
from itertools import izip

def Load_mapper(mapperfile):
    mapper = {}
    with open(mapperfile,'r') as f:
         for x in f:
             line = x.strip().split()
             word = line[0].lower()
             vec = [float(item) for item in line[1:]]
             mapper[word] = vec
    return mapper

def outputHDF5(data,label,filename,labelname='label',dataname='data'):
    print 'data shape: ',data.shape
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    label = [[x.astype(np.float32)] for x in label]
    with h5py.File(filename, 'w') as f:
        f.create_dataset(dataname, data=data, **comp_kwargs)
        f.create_dataset(labelname, data=label, **comp_kwargs)

def reverseComplement(sequence):
    sequence_re = sequence[::-1]
    temp = sequence_re
    for index in range(len(temp)):
        if temp[index] == 'a': sequence_re[index] = 't'
        elif temp[index] == 'c': sequence_re[index] = 'g'
        elif temp[index] == 'g': sequence_re[index] = 'c'
        elif temp[index] == 't': sequence_re[index] = 'a'
        else: sequence_re[index] = 'n'

    return (sequence_re)

def embed_kmer(seq,mapper,instance_len,instance_stride,kernelsize,reverse,kmer):
    instance_num = int((len(seq)-instance_len)/instance_stride) + 1
    bag = []
    for i in range(instance_num):
        instance_fw_kmer = [];instance_bw_kmer = []
        instance_fw = seq[i*instance_stride:i*instance_stride+instance_len]
        if len(instance_fw) < instance_len:
           print >> sys.stderr, "the length of instance is not consistent."; sys.exit(1)
        #process the forward seq
        for j in range(instance_len-kmer+1):
            multinucleotide = instance_fw[j:j+kmer]
            instance_fw_kmer.append(''.join(multinucleotide))
        if reverse:
           instance_bw = reverseComplement(instance_fw)
           #process the forward seq
           for j in range(instance_len-kmer+1):
               multinucleotide = instance_bw[j:j+kmer]
               instance_bw_kmer.append(''.join(multinucleotide))
           instance = instance_fw_kmer + ['n']*kernelsize + instance_bw_kmer
        else:
           instance = instance_fw_kmer
        bag.append(instance)
    mat = np.asarray([mapper[element] if element in mapper else [0.]*pow(4, kmer) for element in bag[0]])
    result = np.asarray([mat])
    for instance in bag[1:]:
        mat = np.asarray([mapper[element] if element in mapper else [0.]*pow(4, kmer) for element in instance])
        result1 = np.asarray([mat])
        result = np.concatenate((result,result1),axis = 0)
    return result

def embed(seq,mapper,instance_len,instance_stride,kernelsize,reverse):

    instance_num = int((len(seq)-instance_len)/instance_stride) + 1
    bag = []
    for i in range(instance_num):
        instance_fw = seq[i*instance_stride:i*instance_stride+instance_len]
        if len(instance_fw) < instance_len:
           print >> sys.stderr, "the length of instance is not consistent."; sys.exit(1)
        if reverse:
           instance_bw = reverseComplement(instance_fw)
           instance = instance_fw + ['n']*kernelsize + instance_bw
        else:
           instance = instance_fw
        bag.append(instance)
    mat = np.asarray([mapper[element] for element in bag[0] if element in mapper])
    result = np.asarray([mat])
    for instance in bag[1:]:
        mat = np.asarray([mapper[element] for element in instance if element in mapper])
        result1 = np.asarray([mat])
        result = np.concatenate((result,result1),axis = 0)
    return result

def seq2feature(data,mapper,label,out_filename,labelname,dataname,instance_len,instance_stride,kernelsize,reverse,kmer=1):
    out = []
    for seq in data:
        if kmer == 1:
           result = embed(seq,mapper,instance_len,instance_stride,kernelsize,reverse)
        else:
           result = embed_kmer(seq,mapper,instance_len,instance_stride,kernelsize,reverse,kmer)
        out.append(result)
    outputHDF5(np.asarray(out),label,out_filename,labelname,dataname)

def convert(pos_file,neg_file,outfile,mapper,labelname,dataname,instance_len,instance_stride,kernelsize,reverse,kmer,run):
    
    with open(pos_file) as posf, open(neg_file) as negf:
        pos_data = posf.readlines(); neg_data = negf.readlines()
    
    pos_seqs = []; neg_seqs = []
    # process posfile and negfile
    for line in pos_data:
        if '>' not in line:
           fw_seq = list(line.strip().lower())
           pos_seqs.append(fw_seq)
    for line in neg_data:
        if '>' not in line: 
           fw_seq = list(line.strip().lower())
           neg_seqs.append(fw_seq)
        
    seqs = pos_seqs + neg_seqs
    # generate their corresponding labels
    label = np.asarray([1] * len(pos_seqs) + [0] * len(neg_seqs))
    if run == 'ws':
       seq2feature(seqs,mapper,label,outfile,labelname,dataname,instance_len,instance_stride,kernelsize,reverse,kmer)
    else:
       seqs_vector = []    
       if kmer == 1:
          for seq in seqs:
              mat = [ mapper[element] for element in seq if mapper.has_key(element)]
              seqs_vector.append(mat)
       else:
          for seq in seqs:
              fw_kmer = [ ''.join(seq[j:j+kmer]) for j in range(len(seq)-kmer+1)]
              mat = [ mapper[element] for element in fw_kmer if mapper.has_key(element)]
              seqs_vector.append(mat)
               
       seqs_vector = np.asarray(seqs_vector)
       outputHDF5(seqs_vector, label, outfile)
       
def parse_args():
    parser = argparse.ArgumentParser(description="Convert sequence and target for Caffe")
    user = pwd.getpwuid(os.getuid())[0]

    # Positional (unnamed) arguments:
    parser.add_argument("posfile",  type=str, help="Sequence in FASTA/TSV format (with .fa/.fasta or .tsv extension)")
    parser.add_argument("negfile",  type=str,help="Label of the sequence. One number per line")
    parser.add_argument("outfile",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.hdf5). ")

    parser.add_argument("-m", "--mapperfile", dest="mapperfile", default="", help="A file mapping each nucleotide to a vector.")
    parser.add_argument("-l", "--labelname", dest="labelname",default='label', help="The group name for labels in the HDF5 file")
    parser.add_argument("-d", "--dataname", dest="dataname",default='data', help="The group name for data in the HDF5 file")
    parser.add_argument("-c", "--instance_len", dest="instance_len", type=int, default=100, help="The length of instance")
    parser.add_argument("-s", "--instance_stride", dest="instance_stride", type=int, default=20, help="The stride of getting instance")
    parser.add_argument("-kernel", "--kernelsize", dest="kernelsize", type=int, default=24, help="The stride of getting instance")
    parser.add_argument('--reverse', dest='reverse', action='store_true', help='build the reverse complement.')
    parser.add_argument('--no-reverse', dest='reverse', action='store_false', help='not to build the reverse complement.')
    parser.add_argument("-kmer", "--kmer", dest="kmer", type=int, default=1, help="the length of kmer")
    parser.add_argument("-run", "--run", dest="run", type=str, default='ws', help="order")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    outdir = dirname(args.outfile)
    if not exists(outdir):
        makedirs(outdir)

    if args.mapperfile == "":
        print 'using ont-hot encoding.'
        args.mapper = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1],'n':[0,0,0,0]}
    else:
        print 'using {} enconding'.format(args.mapperfile)
        args.mapper = Load_mapper(args.mapperfile)
    
    convert(args.posfile,args.negfile,args.outfile,args.mapper,args.labelname,args.dataname,
            args.instance_len,args.instance_stride,args.kernelsize,args.reverse,args.kmer,args.run)
