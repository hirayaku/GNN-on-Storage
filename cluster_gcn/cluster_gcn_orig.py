import argparse
import sys, os
import time
import random
import tqdm

import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import dgl
from dgl.data.utils import get_download_dir

from modules import GraphSAGE
from sampler import ClusterIter
import utils


def main(args):
    # set rand seed: https://stackoverflow.com/a/5012617
    rnd_seed = random.randrange(sys.maxsize)
    random.seed(rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    multitask_data = set(['ppi'])
    multitask = args.dataset in multitask_data

    # load and preprocess dataset
    data = utils.load_data(args)
    g = data.g
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']
    feats = data.features
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0]

    # Normalize features
    if args.normalize and not args.feat_mmap:
        train_feats = feats[train_mask]
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_feats.data.numpy())
        features = scaler.transform(feats.data.numpy())
        feats = torch.FloatTensor(features)

    in_feats = feats.shape[1]
    n_classes = data.num_classes
    n_edges = g.number_of_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
            (n_edges, n_classes,
            n_train_samples,
            n_val_samples,
            n_test_samples))
    # create GCN model
    if args.self_loop and not args.dataset.startswith('reddit'):
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        print("adding self-loop edges")
    # metis only support int64 graph
    g = g.long()

    balance_ntypes = train_mask if args.semi_supervised and args.balance_train else None
    cluster_iterator = ClusterIter(args, g, train_nid,
        balance_ntypes=balance_ntypes, return_nodes=False)

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        g = g.int().to(args.gpu)

    print('labels shape:', g.ndata['label'].shape)
    print('features shape:', feats.shape)

    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.use_pp)

    if cuda:
        model.cuda()

    # Loss function
    # if multitask:
    #     print('Using multi-label loss')
    #     loss_f = nn.BCEWithLogitsLoss()
    # else:
    #     print('Using multi-class loss')
    #     loss_f = nn.CrossEntropyLoss()
    loss_f = nn.CrossEntropyLoss(reduction="none")

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # set train_nids to cuda tensor
    if cuda:
        train_nid = train_nid.cuda()
        print("current memory after model before training",
              torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)
    start_time = time.time()
    best_f1 = -1

    # use tensorboard to log training data
    logger = SummaryWriter(comment=f' {args.dataset} p={args.psize} bc={args.batch_clusters}')
    log_dir = logger.log_dir

    for epoch in range(args.n_epochs):
        total_loss = 0
        for j, cluster in enumerate(tqdm.tqdm(cluster_iterator)):
            model.train()
            cluster_g = cluster
            cluster_h = cluster_g.ndata['feat']
            # forward
            batch_pred = model(cluster_g, cluster_h)
            batch_labels = cluster_g.ndata['label']
            batch_train_mask = cluster_g.ndata['train_mask']
            loss = loss_f(batch_pred[batch_train_mask],
                          batch_labels[batch_train_mask])
            total_loss += loss.sum().item()
            loss = torch.mean(loss)
            # if we allow training nodes to appear more than once in different batches, the following is necessary
            # loss = torch.mean(loss / cluster_g.ndata['count'][batch_train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("Train/loss-minibatch", loss, epoch * len(cluster_iterator) + j)

        logger.add_scalar("Train/loss", total_loss / n_train_samples, epoch)

        # evaluate
        if epoch % args.val_every == args.val_every - 1:
            val_f1_mic, val_f1_mac, val_loss = utils.evaluate(
                model, g, utils.to_torch_tensor(feats), labels, val_mask, multitask)
            print(
                "Val F1-micro {:.4f}, Val F1-macro {:.4f}". format(val_f1_mic, val_f1_mac))
            if val_f1_mic > best_f1:
                best_f1 = val_f1_mic
                print('new best val f1-micro: {:.4f}'.format(best_f1))
                torch.save(model.state_dict(), os.path.join(
                    log_dir, 'best_model.pkl'))
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/F1-micro", val_f1_mic, epoch)

    end_time = time.time()
    print(f'training using time {end_time-start_time}')

    # test
    if args.use_val:
        model.load_state_dict(torch.load(os.path.join(
            log_dir, 'best_model.pkl')))
    test_f1_mic, test_f1_mac, test_loss = utils.evaluate(
        model, g, utils.to_torch_tensor(feats), labels, test_mask, multitask)
    print("Test F1-micro {:.4f}, Test F1-macro {:.4f}". format(test_f1_mic, test_f1_mac))

    logger.add_hparams(
        {"psize": args.psize, "bsize": args.batch_clusters,
         "layers": args.n_layers, "hidden": args.n_hidden,
         "dropout": args.dropout, "epochs": args.n_epochs,
         "lr": args.lr, "weight-decay": args.weight_decay,
         "cluster-method": args.cluster_method, "semi-supervised": args.semi_supervised,
         "balanced-train-nodes": args.balance_train, "rnd-seed": rnd_seed},
        {"test accuracy": test_f1_mic,
         "test loss": test_loss })
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--dataset", type=str, default="reddit-self-loop",
                        help="dataset name")
    parser.add_argument("--rootdir", type=str, default=get_download_dir(),
                        help="directory to read dataset from")
    parser.add_argument("--feat-mmap", action='store_true', help="mmap dataset features")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=3e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--batch-clusters", type=int, default=20,
                        help="batch size")
    parser.add_argument("--psize", type=int, default=1500,
                        help="partition number")
    parser.add_argument("--cluster-method", type=str, default="METIS",
                        help="clustering method: METIS, New or Random")
    parser.add_argument("--semi-supervised", action='store_true',
                        help="Enable semi-supervised training by utilizing val/test node features")
    parser.add_argument("--balance-train", action='store_true',
                        help="balance the number of training nodes in each partition")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--log-every", type=int, default=100,
                        help="the frequency to save model")
    parser.add_argument("--val-every", type=int, default=1,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--use-pp", action='store_true',
                        help="whether to use precomputation")
    parser.add_argument("--normalize", action='store_true',
                        help="whether to use normalized feature")
    parser.add_argument("--use-val", action='store_true',
                        help="whether to use validated best model to test")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--note", type=str, default='none',
                        help="note for log dir")

    args = parser.parse_args()

    print(args)

    main(args)
