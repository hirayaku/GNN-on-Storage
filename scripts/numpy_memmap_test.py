#!/usr/bin/python3

import numpy as np
import argparse, os, time
from pyinstrument import Profiler

argparser = argparse.ArgumentParser("numpy memmap microbenchmark")
argparser.add_argument('--dataset', type=str, default='ogbn-products')
argparser.add_argument('--rootdir', type=str, default='dataset')
argparser.add_argument('--rounds', type=int, default=1000)
argparser.add_argument('--nids', type=int, default=300, help='number of node features to be retrieved per round')
args = argparser.parse_args()

dataset_path = os.path.join(args.rootdir, args.dataset.replace('-', '_'))
feat_path = os.path.join(dataset_path, 'feat_feat.npy')
print(f'mmap features from {feat_path}')

feats = np.lib.format.open_memmap(feat_path, mode='r')
print(f'feats.shape={feats.shape}')
num_nodes = feats.shape[0]

all_nids = np.zeros((args.rounds, args.nids), dtype=int)
for i in range(args.rounds):
    all_nids[i,:] = np.random.randint(num_nodes, size=args.nids, dtype=int)

print("retrieving features...")

traffic = 0
profiler = Profiler()
profiler.start()
start = time.time()

for nids in all_nids:
    features = feats[nids]
    traffic += 4 * features.shape[0] * features.shape[1]

end = time.time()

profiler.stop()
print(profiler.output_text(unicode=True, color=True))

print("Effective Traffic (Bytes) = {:,}".format(traffic))
print("Effective BW (MB/s) = {:.2f}".format(traffic/1e6/(end-start)))

