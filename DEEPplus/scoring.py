import pandas as pd
import numpy as np
import os
import math
import argparse
parser = argparse.ArgumentParser(prog='Npy2Txt',description='Converting numpy output to text format' )
parser.add_argument('--dir', type=str, help='directory of numpy output',required=True)
parser.add_argument('--nfeatures',type=int,help='number of features in model',required=True)
parser.add_argument('--windowsize',type=int,help='size of the window (legnth of context plus core bin)',
                    required=False,default=2000)
parser.add_argument('--binsize',type=int,help='size of the core bin',required=False,default=200)
parser.add_argument('--bed',type=str,help='input bed/vcf for variant position',required=True,default=None)
parser.add_argument('--outdir',type=str,help='path of output data',required=False,default="./")
parser.add_argument('--prefix',type=str,help='prefix of the output training dataset',required=False,default="")

parser.add_argument('--header',type=str,help='header file used for annotation',required=False)
args = parser.parse_args()

wdir=args.dir
windowsize=args.windowsize
binsize=args.binsize
filelist=os.listdir(wdir)
nfeatures=args.nfeatures
outdir=args.outdir
input_bed=pd.read_table(args.bed,sep="\t",header=None)
files=os.listdir(wdir)
os.chdir(wdir)
prefix=args.prefix

arr=[]
in_d1,in_d2=input_bed.shape
for i,file in enumerate(files):
    if file.endswith("npy"):
        loc=windowsize-float(file.split('.')[0])
        dist=max(abs(loc-windowsize/2)-binsize/2,0.0)
        weight=10*math.exp(-1*dist/(1*binsize))
        bed=np.load(wdir+file,allow_pickle=True)
        arr.append(bed[:,in_d2+1:-1]*weight)        
arr=np.stack(arr).mean(0)
print(in_d1,in_d2)
ref_pos,alt_pos=arr[:in_d1,:nfeatures*2],arr[:in_d1,nfeatures*2:]
ref_neg,alt_neg=arr[in_d1:,:nfeatures*2],arr[in_d1:,nfeatures*2:]
diff=(alt_pos+alt_neg)-(ref_pos+ref_neg)
lfc=np.log1p((alt_pos+alt_neg).astype('float')/10/2)-np.log1p((ref_pos+ref_neg).astype('float')/10/2)
ref=(ref_pos+ref_neg)/2
output=pd.DataFrame()
for i in range(nfeatures):
    output['ref_mean_feature{k}'.format(k=str(i+1))]=ref[:,2*i+1]
    output['DEEP_feature{k}'.format(k=str(i+1))]=diff[:,2*i+1]
    output['lfc_feature{k}'.format(k=str(i+1))]=lfc[:,2*i+1]
output=pd.concat((input_bed,output),1)
output.to_csv(outdir+prefix+"DEEP+.txt",sep="\t",index=False)

