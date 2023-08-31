import os, tempfile
import warnings

TMP_DIR = tempfile.gettempdir()
SCRATCH_DIR = None
_ENV_NAMES = ('SCRATCH', 'SCRATCH_DIR', 'DATASETS')
for key in _ENV_NAMES:
    if SCRATCH_DIR is None:
        try:
            SCRATCH_DIR = os.environ[key]
        except KeyError:
            pass
if SCRATCH_DIR is None:
    SCRATCH_DIR = os.getcwd()
    warnings.warn(f"SCRATCH_DIR set to {SCRATCH_DIR}; mind your disk quota", stacklevel=2)
os.environ['SCRATCH_DIR'] = SCRATCH_DIR

def iterable(obj) -> bool:
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

import torch
try:
   from torch_geometric.utils import index_sort as sort
except ImportError:
   from torch import sort

from contextlib import contextmanager
@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

@contextmanager
def parallelism(factor):
    old_num = torch.get_num_threads()
    num_par = max(int(old_num * factor), 1)
    torch.set_num_threads(num_par)
    try:
        yield
    finally:
        torch.set_num_threads(old_num)

import psutil
def get_affinity():
    p = psutil.Process()
    return p.cpu_affinity()
def set_affinity(cpus: list[int]):
    p = psutil.Process()
    p.cpu_affinity(cpus)
def mem_usage():
    p = psutil.Process()
    mem = p.memory_info().rss
    for child in p.children(recursive=True):
        try:
            mem += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return mem / 1e6
def report_mem(point=""):
    mem = mem_usage()
    print(point, f'mem={mem:.2f} MB')

from scipy.stats import wasserstein_distance
def emd(input, target, n_classes):
    def label_histc(labels, n_classes):
        histc = torch.histc(labels.flatten().float(), bins=n_classes, min=0, max=n_classes)
        return histc/histc.sum()
    input_hist, target_hist = label_histc(input, n_classes), label_histc(target, n_classes)
    return wasserstein_distance(input_hist.numpy(), target_hist.numpy())

