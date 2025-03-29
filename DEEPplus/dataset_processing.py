import math
import numpy as np
import pyfasta
import pandas as pd
from utils import *
import argparse
parser = argparse.ArgumentParser(prog='Feature2Sequence',description='Create training dataset for DEEPplus from bed')
parser.add_argument('--bed', type=str, help='input bed file',required=True)
parser.add_argument('--genome',type=str,help='path to reference genome',required=True)
parser.add_argument('--windowsize',type=int,help='size of the window (legnth of context plus core bin)',required=True)
parser.add_argument('--binsize',type=int,help='size of the core bin',required=True)
parser.add_argument('--testchr',type=str,help='chromosomes to be assigned as test dataset')
parser.add_argument('--outdir',type=str,help='path of output data',required=True)
parser.add_argument('--prefix',type=str,help='prefix of the output training dataset',required=True)
args = parser.parse_args()

bedfile=args.bed

dir=args.outdir
prefix=args.prefix
bed_all=pd.read_csv(bedfile, sep='\t', header=None, comment='#')
genome=pyfasta.Fasta(args.genome)
windowsize=args.windowsize
binsize=args.binsize
chrom=args.testchr
CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9','chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17','chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX','chrY']

train_bed=bed_all[bed_all[0]!=chrom]
test_bed=bed_all[bed_all[0]==chrom]
train_cov=((train_bed.iloc[:,3:]/binsize)>=0.5).astype('int8').values
test_cov=((test_bed.iloc[:,3:]/binsize)>=0.5).astype('int8').values
context_len=int((windowsize-binsize)/2)
seqlist=[]
for i in range(train_bed.shape[0]):
    seq = genome.sequence({'chr':train_bed.iloc[i, 0], 'start': train_bed.iloc[i, 1]-context_len, 'stop': train_bed.iloc[i, 2]+context_len})
    seqlist.append(seq)
train_encoded=encodeSeqs(seqlist,inputsize=windowsize).astype('int8')
train_cov=np.concatenate([train_cov,train_cov],axis=0).astype('int8')
seqlist=[]
for i in range(test_bed.shape[0]):
    seq = genome.sequence({'chr': test_bed.iloc[i, 0], 'start': test_bed.iloc[i, 1]-context_len, 'stop': test_bed.iloc[i, 2]+context_len})
    seqlist.append(seq)
test_encoded= encodeSeqs(seqlist, inputsize=windowsize).astype('int8')
test_cov=np.concatenate([test_cov,test_cov],axis=0).astype('int8')
np.save(dir+'wo_'+chrom+"_"+prefix+'_encoded.npy',train_encoded)
np.save(dir+'wo_'+chrom+'_'+prefix+'_cov.npy',train_cov)
np.save(dir+chrom+"_"+prefix+'_encoded.npy',test_encoded)
np.save(dir+chrom+'_'+prefix+'_cov.npy',test_cov)
