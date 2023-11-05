import os, os.path as osp
from data.datasets import load
from data.graphloader import serialize, NodePropPredDataset

def transform(name: str, indir: str, outdir: str) -> dict:
    '''
    serialize the dataset into flat binary files
    return a dictionary describing the transformed dataset
    '''
    attr, data, idx = load(name, indir)
    dataset_dict = {
        'attr': attr,
        'data': data,
        'idx': idx,
    }
    serialize_dir = osp.join(outdir, attr['dir_name'])
    return serialize_dir, serialize(dataset_dict, serialize_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="transform datasets into formats accepted by BaselineNodePropPredDataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, required=True,
                        default='ogbn-products', help="dataset name")
    parser.add_argument('--srcdir', type=str, default="/opt/datasets",
                        help="location of datasets in the original format")
    parser.add_argument('--dstdir', type=str, default=os.environ.get('DATASETS', None),
                        help="location of transformed datasets")
    parser.add_argument('--adj-only', action="store_true",
                        help="create adj from the already transformed dataset")
    args = parser.parse_args()

    if args.adj_only:
        dataset_dir = os.path.join(args.dstdir, args.dataset.replace('-', '_'))
    else:
        # flatten the dataset into binary files
        dataset_dir, _ = transform(args.dataset, indir=args.srcdir, outdir=args.dstdir)
    # create csc/csr
    dataset = NodePropPredDataset(
        dataset_dir, mmap=(False,True), formats=('coo', 'csr', 'csc'),
        create_formats=True,
    )
