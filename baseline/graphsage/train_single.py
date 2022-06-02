# This is the baseline GraphSAGE code from GNS repo
from graphsage import *

import sys
sys.path.append(os.path.abspath("../../"))
import utils

from load_graph import *

def run(args, device, data):
    # Unpack data
    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels = data
    in_feats = train_nfeat.shape[1]
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    if args.disk_feat or args.dataset == 'mag240m':
        val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    else:
        val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]

    #test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
    test_nid = th.nonzero(test_g.ndata['test_mask'], as_tuple=True)[0]

    if args.pure_gpu:
        train_nid = train_nid.to('cuda')
        val_nid = val_nid.to('cuda')
        test_nid = test_nid.to('cuda')

    # Create PyTorch DataLoader for constructing blocks
    #print("setup sampler")
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.DataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    utils.using("Start training")
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        profiler = Profiler()
        profiler.start()
        tic = time.time()
        tic_step = time.time()
        # Loop over the dataloader to sample MFG as a list of blocks
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            #print("Epoch {:d} Step {:d}".format(epoch, step))
            # Load the input features as well as output labels
            #print("dtype of train_labels is : ", train_labels.dtype)
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
            #print("dtype of batch_inputs is : ", batch_inputs.dtype)
            #print("dtype of batch_labels is : ", batch_labels.dtype)
            if batch_inputs.dtype != th.float32:
                batch_inputs = batch_inputs.float()
            if args.disk_feat:
                batch_labels = batch_labels.reshape(-1,)
            blocks = [block.int().to(device) for block in blocks]

            #print("Compute loss and prediction")
            batch_pred = model(blocks, batch_inputs)
            #print("dtype of batch_pred is : ", batch_pred.dtype)
            loss = loss_fcn(batch_pred, batch_labels)
            #loss = F.cross_entropy(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                if len(iter_tput) > 3:
                    print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                        epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()
        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))

        if epoch >= 1:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device, args.batch_size, args.num_workers)
            print('Eval Acc {:.4f}'.format(eval_acc))

    test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device, args.batch_size, args.num_workers)
    print('Test Acc: {:.6f}'.format(test_acc))
    print('Avg epoch time: {:.5f}'.format(avg / epoch))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("single-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--rootdir', type=str, default='../../dataset/')
    argparser.add_argument('--num-epochs', type=int, default=2)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='15,10,5')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument('--disk-feat', action='store_true', help="Put features on disk")
    argparser.add_argument('--pure-gpu', action='store_true',
                           help='Perform both sampling and training on GPU.')

    args = argparser.parse_args()
    print(args)

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    dataset_dir = osp.join(args.rootdir, args.dataset)
    # load feature from file
    feat_shape_file = osp.join(dataset_dir, "feat.shape")
    shape = tuple(utils.memmap(feat_shape_file, mode='r', dtype='int64', shape=(2,)))
    feat_file = osp.join(dataset_dir, "feat.feat")
    print(f"Loading feature data from {feat_file}")
    if not args.disk_feat:
        count = 1
        for s in shape:
            count *= s
        if args.dataset == 'mag240m':
            array = np.fromfile(feat_file, dtype='float16', count=count).reshape(shape)
        else:
            array = np.fromfile(feat_file, dtype='float32', count=count).reshape(shape)
        node_features = th.from_numpy(array)
    else:
        if args.dataset == 'mag240m':
            node_features = utils.memmap(feat_file, random=True, mode='r', dtype='float16', shape=shape)
        else:
            node_features = utils.memmap(feat_file, random=True, mode='r', dtype='float32', shape=shape)
    print("Features: ", node_features.shape)
    feat_len = node_features.shape[1]

    utils.using("Features loaded")

    print(f"Loading graph data from {dataset_dir}")
    graphs, _ = dgl.load_graphs(osp.join(dataset_dir, "graph.dgl"))
    g = graphs[0]
    nv = g.number_of_nodes()
    ne = g.number_of_edges()
    if 'valid_mask' in g.ndata:
        g.ndata['val_mask'] = g.ndata.pop('valid_mask')
    for k, v in g.ndata.items():
        if k.endswith('_mask'):
            g.ndata[k] = v.bool()
    labels = g.ndata.pop('label').flatten()
    n_classes = len(th.unique(labels[th.logical_not(th.logical_or(th.isnan(labels), th.eq(labels, -1)))]))
    labels = labels.long()
    if args.dataset == 'ogbn-papers100M':
        assert n_classes == 172
    elif args.dataset == 'mag240m':
        assert n_classes == 153

    utils.using("Graph loaded")
    print('|V|: {}, |E|: {}, #classes: {}, feat_length: {}'.format(nv, ne, n_classes, feat_len))

    if args.pure_gpu:
        g = g.to('cuda') 
        node_features = node_features.to('cuda')

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
        train_nfeat = train_g.ndata.pop('feat')
        val_nfeat = val_g.ndata.pop('feat')
        test_nfeat = test_g.ndata.pop('feat')
        train_labels = train_g.ndata.pop('label')
        val_labels = val_g.ndata.pop('label')
        test_labels = test_g.ndata.pop('label')
    else:
        train_g = val_g = test_g = g
        train_nfeat = val_nfeat = test_nfeat = node_features
        train_labels = val_labels = test_labels = labels

    #  if not args.data_cpu:
    #      train_nfeat = train_nfeat.to(device)
    #      train_labels = train_labels.to(device)

    print("Create csr/coo/csc formats")
    train_g.formats(['coo', 'csc'])
    val_g.formats(['coo', 'csc'])
    test_g.formats(['coo', 'csc'])
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
           val_nfeat, val_labels, test_nfeat, test_labels
    utils.using("Graph format")

    run(args, device, data)
