import os,json,argparse
import os.path as osp
import torch, dgl
import datasets
import utils

parser = argparse.ArgumentParser(description='coo to csc converter',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
args = parser.parse_args()

def tensor_from_dict(dict, root):
    full_path = osp.join(root, dict['path'])
    shape = dict['shape']
    size = torch.prod(torch.LongTensor(shape)).item()
    dtype = utils.torch_dtype(dict['dtype'])
    return torch.from_file(full_path, size=size, dtype=dtype).reshape(shape)

def edge_index_to_csc(num_nodes, edge_index):
    g = dgl.graph(data=(edge_index[0], edge_index[1]), num_nodes=num_nodes)
    row_ptr, col_idx, eids = g.adj_sparse('csc')
    return row_ptr, col_idx, eids

dir_name = '_'.join(args.dataset.split('-'))
data_dir = osp.join(os.environ['DATASETS'], 'gnnos')
root = osp.join(data_dir, dir_name)
with open(osp.join(root, 'metadata.json')) as f_meta:
    meta_info = json.load(f_meta)

num_nodes = meta_info['num_nodes']
edge_index = tensor_from_dict(meta_info['graph']['edge_index'], root)
print("Generating csc...")
row_ptr, col_idx, _ = edge_index_to_csc(num_nodes, edge_index)
print("Done")

new_dict = { 'format': 'csc' }
print("Serialize csc...")
with utils.cwd(root):
    new_dict['row_ptr'] = datasets.tensor_serialize(row_ptr.numpy(), 'row_ptr')
    new_dict['col_idx'] = datasets.tensor_serialize(col_idx.numpy(), 'col_idx')
    meta_info['graph'] = new_dict
    with open('metadata_csc.json', 'w') as f_meta:
        f_meta.write(json.dumps(meta_info, indent=4, cls=utils.DtypeEncoder))
print("Done")

