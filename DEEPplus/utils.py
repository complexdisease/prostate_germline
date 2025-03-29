import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import random
import pyfasta

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_weights_for_balanced_classes(dataset,nfeatures, nclasses=2):
    mat=np.power(nclasses,np.arange(nfeatures))
    label=dataset.numpy().dot(mat[:,None])
    N=dataset.size(0)
    lb,cts=np.unique(label,return_counts=True)
    weight=[]
    for i,val in enumerate(label):
        idx=np.where(lb==val)
        w=N/(cts[idx]+1)
        weight.append(w)
    weight=np.asarray(weight).flatten()
    return weight

def one_hot(labels:torch.Tensor,num_classes:int, eps = 1e-6) -> torch.Tensor:
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}".format(type(labels)))
    n,m = labels.size(0),label.size(1)
    onehot = torch.zeros((n, num_classes))
    return onehot.scatter_(2,labels.unsqueeze(2), 1.0) + eps

class TrainData(Data.Dataset):
    def __init__(self, seq_file,label_file,root_dir):
        self.seq_data=torch.from_numpy(np.load(root_dir+seq_file))
        self.label_data=torch.from_numpy(np.load(root_dir+label_file))
        self.root_dir=root_dir
    def __len__(self):
        return len(self.seq_data)
    def __getitem__(self, idx):
        data=(self.seq_data[idx],self.label_data[idx])
        return data

def encodeSeqs(seqs, inputsize):
    # Create a lookup table as an array instead of using dictionary lookups
    bases = np.array([[1, 0, 0, 0],  # A
                      [0, 1, 0, 0],  # G
                      [0, 0, 1, 0],  # C
                      [0, 0, 0, 1],  # T
                      [0, 0, 0, 0]]) # N or any other character

    # Create a mapping from character to index in the lookup table
    char_to_index = np.zeros(128, dtype=int)
    char_to_index[ord('A')] = 0
    char_to_index[ord('G')] = 1
    char_to_index[ord('C')] = 2
    char_to_index[ord('T')] = 3
    char_to_index[ord('N')] = 4
    # Lowercase mappings
    char_to_index[ord('a')] = 0
    char_to_index[ord('g')] = 1
    char_to_index[ord('c')] = 2
    char_to_index[ord('t')] = 3
    char_to_index[ord('-')] = 4

    # Pre-allocate the array for all sequences
    num_seqs = len(seqs)
    seqsnp = np.zeros((num_seqs, 4, inputsize), dtype=np.float32)

    # Iterate over sequences
    for i, seq in enumerate(seqs):
        # Truncate or pad the sequence to fit the input size
        cline = seq[(len(seq) - inputsize) // 2 : (len(seq) + inputsize) // 2]

        # Convert sequence characters to indices
        indices = np.frombuffer(cline.encode('ascii'), dtype=np.uint8)
        indices = char_to_index[indices]

        # Map the indices to one-hot encoding using NumPy's advanced indexing
        seqsnp[i, :, :len(cline)] = bases[indices].T

    # Create reverse complement by flipping both dimensions
    seqsnp_flipped = seqsnp[:, ::-1, ::-1]

    # Concatenate original and flipped sequences along the first axis
    seqsnp = np.concatenate([seqsnp, seqsnp_flipped], axis=0)

    return seqsnp


