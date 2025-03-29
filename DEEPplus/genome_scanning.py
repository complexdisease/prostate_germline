import math
import sys
import pyfasta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import argparse
from model import DEEPPLUS
from utils import *
parser = argparse.ArgumentParser(description='genome scanning for lists of snps')
parser.add_argument('--model', type=str, help='path to model',required=True)
parser.add_argument('--tsv', type=str, help='path to variants bed',required=True)
parser.add_argument("-g",type=str,help="path to genome fasta", required=True, default=None)
parser.add_argument("--output",type=str,help="output directory",required=True)
parser.add_argument('--nfeatures',"-N",type=int,help='number of features in model',required=True,default=1)
parser.add_argument("--batchsize","-b",type=int,help="Batchsize",required=False,default=500)
parser.add_argument("--windowsize",type=int,help="size of window",required=False,default=2000)
parser.add_argument("--stepsize",type=int,help="step size for genome scanning",required=False,default=20)
parser.add_argument("--start",type=int,help="start position for scanning",required=False,default=900)
parser.add_argument("--end",type=int,help="end position for genome scanning",required=False,default=1100)
args = parser.parse_args()
eps=1E-6
start,end=args.start,args.end+1
bedfile=args.tsv
dir=args.output
bed=pd.read_csv(bedfile, sep='\t', header=None, comment='#')
bed.iloc[:, 0] = 'chr' + bed.iloc[:, 0].map(str).str.replace('chr', '')
Batchsize=args.b
nfeatures=args.N
genome=pyfasta.Fasta(args.g)
windowsize=args.windowsize
stepsize=args.stepsize
usedbed=bed

cnn=DEEPPLUS([1,1,1,1],num_classes=2*nfeatures,expansion=4)
cnn.load_state_dict(torch.load(args.model))
cnn.eval()
if torch.cuda.is_available():
    torch.cuda.get_device_name(0)
    cnn.cuda()

for shift in range(start,end,stepsize):
    print("Genome scanning on step (bp):", shift)
    ref_seqlist, mut_seqlist ,match_ref_list = [], [], []
    distance=shift-windowsize/2
    for i in range(usedbed.shape[0]):
        seq = genome.sequence({'chr': usedbed.iloc[i, 0], 'start': usedbed.iloc[i, 1]-windowsize+shift, 'stop': usedbed.iloc[i, 1]+shift})
        mutpos = windowsize - shift
        ref,alt=usedbed.iloc[i,2],usedbed.iloc[i,3]
        mutseq = seq[:mutpos]+alt+seq[mutpos+len(ref):]
        refseq = seq[:mutpos]+ref+seq[mutpos+len(ref):]
        match_ref=seq[mutpos:(mutpos + len(ref))].upper() == ref.upper()

        ref_seqlist.append(refseq)
        mut_seqlist.append(mutseq)
        match_ref_list.append(match_ref)
    if shift == 0:
        print('Matched REF:\t', np.sum(match_ref_list), 'Total Sites:\t', len(match_ref_list))

    ref_encoded = encodeSeqs(ref_seqlist, inputsize=windowsize).astype(np.float32)
    alt_encoded = encodeSeqs(mut_seqlist, inputsize=windowsize).astype(np.float32)
    
    ref_preds = []
    for i in range(int(1 + (ref_encoded.shape[0] - 1) / Batchsize)):
        input = torch.from_numpy(ref_encoded[int(i * Batchsize):int((i + 1) * Batchsize), :, :]).cuda()
        ref_preds.append(cnn.forward(input).cpu().detach().view(-1,2*nfeatures).numpy().copy())
    ref_preds = np.vstack(ref_preds)

    alt_preds = []
    for i in range(int(1 + (alt_encoded.shape[0] - 1) / Batchsize)):
        input = torch.from_numpy(alt_encoded[int(i * Batchsize):int((i + 1) * Batchsize), :, :]).cuda()
        alt_preds.append(cnn.forward(input).cpu().detach().view(-1,2*nfeatures).numpy().copy())
    alt_preds = np.vstack(alt_preds)
    
    output=np.concatenate((ref_preds,alt_preds),axis=1)
    match_ref_list=np.array(match_ref_list).reshape(-1,1)
    match_ref_list=np.vstack((match_ref_list,match_ref_list))
    output=np.concatenate((output,match_ref_list),axis=1)
    pos,neg=usedbed,usedbed
    pos['strand']='pos'
    neg['strand']='neg'
    stack=pd.concat((pos,neg),axis=0,ignore_index=True).to_numpy()
    stack=np.concatenate((stack,output),axis=1)
    np.save(dir+str(shift)+'.npy',stack)

