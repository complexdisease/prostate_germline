# DEEPplus model training

Training DEEPplus model from consensus ATAC-seq/ChIP-seq peaks

01 Creating coverage files from called peaks

preprocessing.sh -o [outdir] -g [reference genome chromsizes] [peak files in bed]

02 Generating training and testing datasets from step 01

dataset_processing.py --bed [output bed from step 01] --genome [reference fasta] --testchr [chromosomes split for testing dataset] --outdir [Output directories] --prefix [Prefix for output dataset]

03 DEEPplus model training

train.py --nfeatures [number of features types from input] --cuda [GPU device to use] --lr [learning rate] --nepoch [number of epochs for training] --batchsize [batchsize for training] --train_prefix [prefix of the training dataset] --train_path [directory of the training dataset] --test_prefix [prefix of the training dataset] --test_path [directory of the training dataset] --val_pct [percentage of datasets for validation] --out [path of the output models]

04 In silico mutagenesis with DEEPplus

genome_scanning.py --nfeatures [number of features types from input] --batchsize [batchsize for training] -g [reference fasta] --model [path of the model from 03] --tsv [variants for genome scanning in bed/tsv format] --output [directory storing scanning results]

05 Scoring variants

scoring.py --nfeatures [number of features types from input] --dir [directory storing scanning results of 04] --bed [variants for genome scanning in bed/tsv format] --outdir [path of final scores]
