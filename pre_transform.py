import os
from data.datasets import transform
from data.graphloader import NodePropPredDataset

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
    args = parser.parse_args()

    # flatten the dataset into binary files
    dataset_dir, _ = transform(args.dataset, indir=args.srcdir, outdir=args.dstdir)
    # create csc/csr
    dataset = NodePropPredDataset(
        dataset_dir, mmap=(False,True), formats=('coo', 'csr', 'csc'),
        create_formats=True,
    )
