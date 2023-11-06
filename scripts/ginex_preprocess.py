import argparse
from data.graphloader import NodePropPredDataset
from data.io import Dtype
import scipy
import numpy as np
import json
import torch
import os


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
args = argparser.parse_args()

# Download/load dataset
print('Loading dataset...')
dataset = NodePropPredDataset(
    os.path.join('/mnt/md0/hb_datasets', args.dataset.replace('-','_')), mmap=True)
root = '/mnt/md0/datasets'
os.makedirs(root, exist_ok=True)
dataset_path = os.path.join(root, args.dataset + '-ginex')
print('Done!')

# Construct sparse formats
print('Creating coo/csc/csr format of dataset...')
num_nodes = dataset[0].num_nodes
csc = dataset[0].adj_t
csr = dataset[0].adj
print('Done!')

# Save csc-formatted dataset
indptr, indices, _ = csc.csr()
indptr = indptr.numpy()
indices = indices.numpy()
features = dataset[0].x.numpy()
labels = dataset[0].y.numpy()

os.makedirs(dataset_path, exist_ok=True)
indptr_path = os.path.join(dataset_path, 'indptr.dat')
indices_path = os.path.join(dataset_path, 'indices.dat')
features_path = os.path.join(dataset_path, 'features.dat')
labels_path = os.path.join(dataset_path, 'labels.dat')
conf_path = os.path.join(dataset_path, 'conf.json')
split_idx_path = os.path.join(dataset_path, 'split_idx.pth')

print('Saving indptr...')
indptr_mmap = np.memmap(indptr_path, mode='w+', shape=indptr.shape, dtype=indptr.dtype)
indptr_mmap[:] = indptr[:]
indptr_mmap.flush()
print('Done!')

print('Saving indices...')
indices_mmap = np.memmap(indices_path, mode='w+', shape=indices.shape, dtype=indices.dtype)
indices_mmap[:] = indices[:]
indices_mmap.flush()
print('Done!')

features_mmap = dataset[0].x.numpy()
#  print('Saving features...')
#  features_mmap = np.memmap(features_path, mode='w+', shape=dataset[0].x.shape, dtype=np.float32)
#  features_mmap[:] = features[:]
#  features_mmap.flush()
#  print('Done!')

print('Saving labels...')
labels = labels.astype(np.float32)
labels_mmap = np.memmap(labels_path, mode='w+', shape=dataset[0].y.shape, dtype=np.float32)
labels_mmap[:] = labels[:]
labels_mmap.flush()
print('Done!')

print('Making conf file...')
mmap_config = dict()
mmap_config['num_nodes'] = dataset[0].num_nodes
mmap_config['indptr_shape'] = tuple(indptr.shape)
mmap_config['indptr_dtype'] = str(indptr.dtype)
mmap_config['indices_shape'] = tuple(indices.shape)
mmap_config['indices_dtype'] = str(indices.dtype)
mmap_config['indices_shape'] = tuple(indices.shape)
mmap_config['indices_dtype'] = str(indices.dtype)
mmap_config['indices_shape'] = tuple(indices.shape)
mmap_config['indices_dtype'] = str(indices.dtype)
mmap_config['features_shape'] = tuple(features_mmap.shape)
mmap_config['features_dtype'] = str(features_mmap.dtype)
mmap_config['labels_shape'] = tuple(labels_mmap.shape)
mmap_config['labels_dtype'] = str(labels_mmap.dtype)
mmap_config['num_classes'] = dataset.num_classes
json.dump(mmap_config, open(conf_path, 'w'))
print('Done!')

print('Saving split index...')
torch.save(dataset.get_idx_split(), split_idx_path)
print('Done!')

# Calculate and save score for neighbor cache construction
print('Calculating score for neighbor cache construction...')
score_path = os.path.join(dataset_path, 'nc_score.pth')
csc_indptr_tensor = csc.csr()[0]
csr_indptr_tensor = csr.csr()[0]

eps = 0.00000001
in_num_neighbors = (csc_indptr_tensor[1:] - csc_indptr_tensor[:-1]) + eps
out_num_neighbors = (csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]) + eps
score = out_num_neighbors / in_num_neighbors
print('Done!')

print('Saving score...')
torch.save(score, score_path)
print('Done!')
