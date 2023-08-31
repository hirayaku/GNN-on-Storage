# %%
import torch, torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.sampler.utils import to_csc
from torch_sparse import SparseTensor

print("PyTorch:", torch.__version__, *torch.__path__)
print("PyG:", pyg.__version__, *pyg.__path__)
print("CPU parallelism:", torch.get_num_threads())

from data.graphloader import NodePropPredDataset
from shuffle import global_batching_dp, hier_batching_dp

dataset = NodePropPredDataset(
    root='/mnt/md0/hb_datasets/ogbn_arxiv',
    mmap=False,
)
assert not dataset.is_directed, "expect the dataset to be undirected"
data = dataset[0]
colptr, row, _ = to_csc(data)
data.adj_t = SparseTensor(rowptr=colptr, col=row, sparse_sizes=data.size())
in_feats = data.x.shape[1]
n_classes = dataset.num_classes
idx = dataset.get_idx_split()
train_nid = idx['train']
val_nid = idx['valid']
test_nid = idx['test']
print(n_classes, "classes,", data)

def label_histc(labels, n_classes):
    histc = torch.histc(labels.flatten().float(), bins=n_classes, min=0, max=n_classes)
    histc[histc==0]=1e-9
    return histc/histc.sum()

def diff_dist(input, target):
    input_hist, target_hist = label_histc(input, n_classes), label_histc(target, n_classes)
    return input_hist - target_hist

def kl_div(input, target):
    input_hist, target_hist = label_histc(input, n_classes), label_histc(target, n_classes)
    loss_pointwise = target_hist * (target_hist.log() - input_hist.log())
    return loss_pointwise.sum() / input.size(0)

from scipy.stats import wasserstein_distance
def emd(input, target):
    input_hist, target_hist = label_histc(input, n_classes), label_histc(target, n_classes)
    return wasserstein_distance(input_hist.numpy(), target_hist.numpy())
    # return (input_hist-target_hist).abs().max().item()

def emd_distr(input, target):
    return wasserstein_distance(input.numpy(), target.numpy())

def simulate_distance(labels, n_blocks: int, batch_blocks:int):
    # generate a list of `n_blocks` samples,
    # following the label distribution of true `labels`
    n_classes = labels.max().item() + 1
    label_dist = label_histc(labels.flatten(), n_classes)
    samples_per_label = (label_dist * n_blocks).round()
    samples = torch.zeros((int(samples_per_label.sum()),), dtype=torch.long)
    start = 0
    current_label = 0
    for cnt in samples_per_label:
        end = start + int(cnt)
        samples[start:end] = current_label
        current_label += 1
        start = end
    # perform shuffling and batching on samples
    dp = global_batching_dp(samples, batch_blocks, shuffle=True)
    distance = torch.tensor([emd(batch, labels) for batch in dp])
    return distance.mean(), distance.std()

from torchdata.datapipes.iter import ShardingFilter, Prefetcher
from trainer.baseline import make_blocks, make_ns_dp, edge_cuts

train_labels = data.y[train_nid]
fanout = [15, 10, 5]
num_blocks = 1024
target_blocks = num_blocks//16

gs_dp = global_batching_dp(train_nid, 1024, shuffle=True)
rand_blocks, ptn = make_blocks(data, train_nid, num_blocks, mode='random')
print(f"rand: cuts={edge_cuts(data, ptn)}") #, {[len(block) for block in rand_blocks]}")
rand_dp = hier_batching_dp(
    rand_blocks, target_blocks, 1024,
    shuffle=True, drop_thres=1/4,
)

metis_blocks, ptn = make_blocks(data, train_nid, num_blocks, mode='metis')
print(f"metis: cuts={edge_cuts(data, ptn)}") #, {[len(block) for block in metis_blocks]}")
metis_dp = hier_batching_dp(
    metis_blocks, target_blocks, 1024,
    shuffle=True, drop_thres=1/4,
)
if train_nid.size(0) < 1e6:
    shuffled = torch.cat(list(metis_dp))
    train_ratio = shuffled.size(0) / train_nid.size(0)
    print("train_ratio:", train_ratio)
    if train_ratio == 1:
        assert (torch.cat(list(metis_dp)).sort()[0] == train_nid.sort()[0]).all()
fennel_blocks, ptn = make_blocks(data, train_nid, num_blocks, mode='fennel')
print(f"fennel: cuts={edge_cuts(data, ptn)}") #, {[len(block) for block in fennel_blocks]}")
fennel_dp = hier_batching_dp(
    fennel_blocks, target_blocks, 1024,
    shuffle=True, drop_thres=1/4,
)

gs_emd = torch.tensor([emd(data.y[batch], data.y[train_nid]) for batch in gs_dp])
rand_emd = torch.tensor([emd(data.y[batch], data.y[train_nid]) for batch in rand_dp.cycle(3)])
metis_emd = torch.tensor([emd(data.y[batch], data.y[train_nid]) for batch in metis_dp.cycle(3)])
fennel_emd = torch.tensor([emd(data.y[batch], data.y[train_nid]) for batch in fennel_dp.cycle(3)])
print("gs batch:", gs_emd.mean(), gs_emd.std())
print("rand HB:", rand_emd.mean(), rand_emd.std())
print("metis HB:", metis_emd.mean(), metis_emd.std())
print("fennel HB:", fennel_emd.mean(), fennel_emd.std())
