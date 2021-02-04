import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import math
import argparse
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
import tqdm
import traceback
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import json
import matplotlib.pyplot as plt
import pdb
from sklearn.metrics import f1_score
epsilon = 1 - math.log(2)


def calc_f1(pred, batch_labels):
   
        pred_labels = pred.detach().cpu().argmax(dim=1)
        

        score = f1_score(batch_labels.cpu(), pred_labels, average="micro")

        return score


# more parameters are needed to contruct SAGE
class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 fanout,
                 buffer_size,
                 batch_size):

        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.fanout = fanout
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    # for importance sampling, use the fanout and dst_in_degrees
    def forward(self, blocks, x):
        h = x

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
          

            h_dst = h[:block.number_of_dst_nodes()]

         
            h = layer(block, (h, h_dst))
            
          
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [60,60,60], args.buffer, args.buffer_size)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    evaluator = Evaluator(name="ogbn-arxiv")

    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]


def evaluate(model, g, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        inputs = g.ndata['feat']
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return calc_f1(pred[val_nid], labels[val_nid]), calc_f1(pred[test_nid], labels[test_nid]), pred


def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
#    pdb.set_trace()

    batch_inputs = g.ndata['feat'][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


#### Entry point
def run(args, device, data):
   
    # added by jialin, additional paramenters
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, g = data
    number_of_nodes = g.number_of_nodes()
    in_degree_all_nodes = g.out_degrees()
    # Define model and optimizer
    # added by jialin, additional paramenters
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout,
                 [int(fanout) for fanout in args.fan_out.split(',')], args.buffer_size,
                 args.batch_size / number_of_nodes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    keys = ['Loss', 'Train Acc', 'Test Acc', 'Eval Acc']
    history = dict(zip(keys, ([] for _ in keys)))
    fanout= [int(fanout) for fanout in args.fan_out.split(',')]
    in_degree_all_nodes = g.out_degrees()
    prob = np.true_divide(1,in_degree_all_nodes)
   

    Adj = g.adjacency_matrix()
    pro = (Adj*Adj*Adj).mm(np.reshape(prob, [len(prob), 1]).float()).numpy()
    pro = np.true_divide(pro, sum(pro))
    pro = np.reshape(pro,[len(pro),])
    
    while pro.sum() != 1:
        pro /= pro.sum()
    
        #eval_acc, test_acc, pred = evaluate(model, g, labels, val_nid, test_nid, args.val_batch_size, device)
    #print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))
    for epoch in range(args.num_epochs):
        if epoch % args.buffer_rs_every == 0:
            if args.buffer_size != 0:
                # initial the buffer
                num_nodes = g.num_nodes()
                num_sample_nodes = int(args.buffer_size * num_nodes)
                
                
                

                while pro.sum() != 1:
                    pro /= pro.sum()
#                prob = np.divide(in_degree_all_nodes, sum(in_degree_all_nodes))
                args.buffer = np.random.choice(num_nodes, num_sample_nodes, replace=False,p=pro)
#                args.buffer = np.random.permutation(num_nodes)[:num_sample_nodes]
            sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [1,1,1],[-1,100,100], args.buffer, args.buffer_size,g)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                train_nid,
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

        tic = time.time()


        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            tic_step = time.time()

            # copy block to gpu
            blocks = [blk.int().to(device) for blk in blocks]

  
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device)
          
                 
            batch_pred = model(blocks, batch_inputs)
            loss = cross_entropy(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = calc_f1(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                        epoch, step, loss.item(), acc, np.mean(iter_tput[3:]), gpu_mem_alloc))

        history['Loss'].append(loss.item())
        history['Train Acc'].append(acc)

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0:
            sampler_test = dgl.dataloading.MultiLayerNeighborSampler(
                                [1,1,1],[1000,1000,1000], args.buffer, args.buffer_size,g)

            test_dataloader = dgl.dataloading.NodeDataLoader(
                g,
                test_nid,
                sampler_test,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)
            model.eval()
            test_acc  =[]
            for input_nodes, seeds, blocks in tqdm.tqdm(test_dataloader):
                            blocks = [blk.int().to(device) for blk in blocks]
                            batch_inputs = g.ndata['feat'][input_nodes].to(device)
                            batch_labels = labels[seeds].to(device)
                            batch_pred = model(blocks, batch_inputs)
                            batch_labels = th.reshape(batch_labels, (-1,))
                            test_acc.append( calc_f1(batch_pred, batch_labels) )

            print('Test Acc {:.4f}'.format( np.mean(test_acc)))
            
            history['Test Acc'].append(test_acc)
           

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    epochs = range(args.num_epochs)
    plt.figure(1)
    plt.plot(epochs, history['Loss'], label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(
        'final/loss_' + args.dataset + '_' + str(args.buffer_size) + '_' + str(args.buffer_rs_every) + '.png')
    plt.show()
    plt.figure(2)
    plt.plot(epochs, history['Train Acc'], label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(
        'final/acc_' + args.dataset + '_' + str(args.buffer_size) + '_' + str(args.buffer_rs_every) + '.png')
    plt.show()

    json_r = json.dumps(history)
    f = open('final/history_' + args.dataset + '_' + str(args.buffer_size) + '_' + str(
        args.buffer_rs_every) +'.json', "w")
    f.write(json_r)
    f.close()
    return best_test_acc


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,10,15')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--val-batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=9)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=8,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--buffer-size', type=float, default=0.01)
    argparser.add_argument('--buffer', type=np.ndarray, default=None)
    argparser.add_argument('--buffer_rs-every', type=int, default=1)
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load data
    data = DglNodePropPredDataset(name="ogbn-products")
#    data = DglNodePropPredDataset(name="ogbn-arxiv")
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    graph, labels = data[0]

    in_feats = graph.ndata['feat'].shape[1]
    n_classes = (labels.max() + 1).item()
    n_edges = graph.number_of_edges()
    n_nodes = graph.number_of_nodes()
#    pdb.set_trace()

    graph.create_formats_()
    # Pack data
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, graph

    # Run 1 times
    test_accs = []
    for i in range(1):
        test_accs.append(run(args, device, data))
        print('Average test accuracy:', np.mean(test_accs), '±', np.std(test_accs))
