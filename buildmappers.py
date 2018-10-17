#!/usr/bin/env python
# encoding: utf-8

import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Build mappers.")
    parser.add_argument('-m', dest='mapperfile', type=str, default="", help="A file storing the mappers")
    parser.add_argument('-k', dest='kmer', type=int, default=1, help='the value of kmer')
    
    return parser.parse_args()


# build a code table
def Buildmapper(kmer):
    length = kmer
    alphabet = ['a','c','g','t']
    mapper = ['']
    while length > 0:
        mapper_len = len(mapper)
        for base in range(mapper_len):
            for letter in alphabet:
                mapper.append(mapper[base] + letter)
        #delete the original conents
        while mapper_len > 0:
            mapper.pop(0)
            mapper_len -= 1

        length -= 1

    code = np.eye(len(mapper), dtype = int)
    encoder = {}
    for i in range(len(mapper)):
        encoder[mapper[i]] = list(code[i,:])
    
    number = pow(len(alphabet), kmer)
    encoder['n'] = [0]*number
    return encoder


def main():
    args = parse_args()
    encoder = Buildmapper(args.kmer)
    with open(args.mapperfile, 'w') as f:
        for key, value in encoder.items():
            content = [str(x) for x in value]
            content = key + ' ' + ' '.join(content)
            print >> f, "%s" % content
    
if __name__ == '__main__': main()    
    
    
    
    
    
    
