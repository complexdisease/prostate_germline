import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import copy
import random
import os
from model import DEEPPLUS
from utils import *
import argparse
parser = argparse.ArgumentParser(prog='DEEPplus training',description='training DEEPplus from scratch')
parser.add_argument('--nfeatures', type=int, help='number of features',required=True, default=1)
parser.add_argument('--dir',type=str,help='working directories',required=False, default="./")
parser.add_argument('--windowsize',type=int,help='size of windows (bp)',required=False,default=2000)
parser.add_argument('--binsize',type=int,help='size of the core bin',required=False,default=200)
parser.add_argument('--cuda',type=int,help='which GPU to use',required=False, default=0)
parser.add_argument('--out',type=str,help='path of the saved model',required=True)
parser.add_argument('--nepoch',type=int,help='number of epochs',required=True,default=5)
parser.add_argument('--lr',type=float,help='learning rate',required=True)
parser.add_argument('--batchsize',type=int,help='batch size for training',required=True,default=500)
parser.add_argument('--train_path',type=str,help='path of the training dataset',required=True,default=None)
parser.add_argument('--train_prefix',type=str,help='prefix of the training dataset',required=True,default=None)

parser.add_argument('--test_path',type=str,help='path of the testing dataset',required=True,default=None)
parser.add_argument('--test_prefix',type=str,help='prefix of the testing dataset',required=True,default=None)
parser.add_argument('--val_pct',type=float,help='percentage of records split for validation',required=True,default=1e-6)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=torch.cuda.get_device_name(args.cuda)
nfeatures=args.nfeatures
wdir=args.dir
EPOCH=args.nepoch
Batchsize=args.batchsize
LR=args.lr
windowsize=args.windowsize
binsize=args.binsize
SEED=42
train_data=TrainData(seq_file=args.train_prefix+"_encoded.npy", label_file=args.train_prefix+"_cov.npy", root_dir=args.train_path)
seed(SEED)
val_len=int(args.val_pct*len(train_data))

train_data,val_data=Data.random_split(train_data, [len(train_data)-val_len,val_len])
weights = make_weights_for_balanced_classes(train_data.dataset.label_data, nfeatures=train_data.dataset.label_data.size(1),nclasses=2)
weights = torch.HalfTensor(weights)
sampler = Data.sampler.WeightedRandomSampler(weights, len(weights))
train_split=Data.Subset(train_data.dataset,train_data.indices)
val_split=Data.Subset(val_data.dataset,val_data.indices)
train_loader=Data.DataLoader(dataset=train_split, batch_size=Batchsize,shuffle=True, num_workers=4,drop_last=True)
val_loader=Data.DataLoader(dataset=val_split, batch_size=Batchsize, shuffle=True, num_workers=4,drop_last=True)
print('number of training data:',len(train_loader),'number of validation data:',len(val_loader))
loss_func=nn.BCELoss(reduction='mean')

torch.set_default_dtype(torch.float32)
train_loss = 0.0
test_loss = 0.0
best_tpr,best_acc,best_ppv,best_bacc=0.0,0.0,0.0,0.0
cnn=DEEPPLUS([1,1,1,1],num_classes=2*nfeatures,expansion=4)
wandb.init(project='deepplus', config={'seed':SEED,'batchsize':Batchsize,'lr':LR,"nfeature":nfeatures,'epochs':EPOCH}, reinit=True)
if args.pretrained is not None:
    cnn.load_state_dict(torch.load(args.pretrained))
if torch.cuda.is_available():
    torch.cuda.get_device_name(args.cuda)
    torch.cuda.empty_cache()
    cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
for epoch in range(EPOCH):
    torch.cuda.empty_cache()
    print("Epoch:\t",epoch)
    cnn.train()
    train_loss,corrects=0.0,0.0

    for step, (x, y) in enumerate(train_loader):
        input=x.float()
        y=torch.zeros(y.shape[0],y.shape[1],2).scatter_(2,torch.as_tensor(y).long().unsqueeze(2),1.0).float()
        label=y.cuda()
        output = cnn(input.cuda())
        
        loss = loss_func(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(epoch,step,loss.item())
            wandb.log({'train/loss': loss.item()})
    print("Waiting Test!")
    with torch.no_grad():
        total=0.0
        tp,allp,allpred,acc=torch.zeros(nfeatures),torch.zeros(nfeatures),torch.zeros(nfeatures),torch.zeros(nfeatures)
        for step,data in enumerate(val_loader):
            cnn.eval()
            val_x,val_y=data
            total+=val_y.size(0)
            output=cnn(val_x.float().cuda())
            val_y=val_y.float()
            pred=torch.max(output,2)[1].float().cpu()
            acc=acc+torch.eq(pred,val_y).int().sum(dim=0)
            tp=tp+torch.sum(torch.mul(pred,val_y),dim=0)
            allp=allp+torch.sum(val_y,dim=0)
            allpred=allpred+torch.sum(pred,dim=0)
    
    accuracy=torch.div(torch.sum(acc,dim=0),1.0*total)
    tn=total-allp.add(allpred).sub(tp)        
    tpr,ppv,tnr=torch.div(tp,allp),torch.div(tp,allpred),tn.div(allpred.sub(tp).add(tn))
    bacc=0.5*(tpr+tnr)
    print("Total,Acc,Accuracy,TP,allp,allpred,TN,TPR,TNR,PPV,BACC:\t",total,acc,accuracy,tp,allp,allpred,tn,tpr,tnr,ppv,bacc)
    '''
    wandb.log({'val/accuracy': accuracy.item(),
                'val/tp rate_1': tpr[0].item(),
                #'val/tp_rate_2': tpr[1].item(),
                'val/ppv_1': ppv[0].item(),
                #'val/ppv_2':ppv[1].item(),
                'val/tnr_1': tnr[0].item(),
                #'val/tnr_2':tnr[1].item(),
                'val/bacc_1': bacc[0].item(),
                #'val/bacc_2':bacc[1].item()
                })
    '''
    if bacc.ge(best_bacc).int().sum().ge(int((nfeatures+1)/2.0)).item():
        best_model = copy.deepcopy(cnn.state_dict())
        best_acc,best_tpr,best_ppv,best_bacc,best_tnr=acc,tpr,ppv,bacc,tnr
        torch.save(cnn.state_dict(),wdir+'/tmp.pth')
    cnn.load_state_dict(best_model)
            
torch.save(cnn.state_dict(), os.path.join(wdir,args.model))

cnn.cpu()
cnn.eval()
test_data=TrainData(seq_file=args.test_prefix+"_encoded.npy", label_file=args.test_prefix+"_cov.npy", root_dir=args.test_path)

test_loader=Data.DataLoader(dataset=test_data, batch_size=Batchsize, shuffle=True, num_workers=1,drop_last=True)
for batch in test_loader:
    seq_test, lb_test = batch
    test_output = cnn(seq_test.float())
    pred=torch.max(test_output,2)[1].float()
    lb_test=lb_test.float()
    add=torch.eq(pred,lb_test).int()
    accuracy=torch.div(torch.sum(add,dim=0),Batchsize*1.0)
    tp=torch.sum(torch.mul(pred,lb_test),dim=0)
    allp=torch.sum(lb_test,dim=0)
    allpred=torch.sum(pred,dim=0)
    test_loss = loss_func(pred.float(), lb_test)
    print("tp:\t",tp,"allp:\t",allp,"allpred:\t",allpred,"accuracy:\t",accuracy)
    print('Test Loss:\t', test_loss)
    '''
    wandb.log({'test/loss': test_loss.item(),
        'test/tp rate_1':tp[0].item()/allpred[0].item(),
        #'test/tp_rate_2':tp[1].item()/allpred[1].item(),
        'test/accuracy': accuracy.item()})
    '''
    break
