#!/usr/bin/python3

import numpy as np
import mmap
import ctypes
import sys, argparse, os, time
from pyinstrument import Profiler

argparser = argparse.ArgumentParser("numpy memmap microbenchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--dataset', type=str, default='ogbn-products')
argparser.add_argument('--rootdir', type=str, default='dataset')
argparser.add_argument('--rounds', type=int, default=1000, help="number of rounds")
argparser.add_argument('--nids', type=int, default=300, help="number of nodes to retrieve per round")
argparser.add_argument('--madvise', action="store_true", help="random access advise on mmap")
args = argparser.parse_args()

dataset_path = os.path.join(args.rootdir, args.dataset.replace('-', '_'))
feat_path = os.path.join(dataset_path, 'feat.feat')
shape_path = os.path.join(dataset_path, 'feat.shape')
print(f'mmap features from {feat_path}')

shape = np.memmap(shape_path, mode='r', dtype='int64')
feats = np.memmap(feat_path, mode='r', dtype="float32", shape=tuple(shape))
if args.madvise:
    print("madvise")
    try:
        feats.madvise(mmap.MADV_RANDOM)
    except AttributeError:
        # in python<3.8 mmap doesn't provide madvise
        # https://github.com/numpy/numpy/issues/13172
        madvise = ctypes.CDLL('libc.so.6', use_errno=True).madvise
        madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        madvise.restype = ctypes.c_int
        if madvise(feats.ctypes.data, feats.size * feats.dtype.itemsize, 1) != 0:
            errno = ctypes.get_errno()
            print(f"madvise failed with error {errno}: {os.strerror(errno)}")
            sys.exit(errno)

print(f'feats.shape={feats.shape}')
num_nodes = feats.shape[0]

all_nids = np.zeros((args.rounds, args.nids), dtype=int)
for i in range(args.rounds):
    all_nids[i,:] = np.random.randint(num_nodes, size=args.nids, dtype=int)

print("retrieving features...")

profiler = Profiler()
profiler.start()
start = time.time()

traffic = 4 * args.rounds * args.nids * feats.shape[1]

for nids in all_nids:
    #  traffic += 4 * features.shape[0] * features.shape[1]
    features = feats[nids]

end = time.time()

profiler.stop()
print(profiler.output_text(unicode=True, color=True))

print("Effective Traffic (Bytes) = {:,}".format(traffic))
print("Effective BW (MB/s) = {:.2f}".format(traffic/1e6/(end-start)))

