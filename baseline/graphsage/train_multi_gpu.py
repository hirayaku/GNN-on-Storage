# This is the baseline GraphSAGE code from DGL
from graphsage import *
import sys
sys.path.append(os.path.abspath("../../"))
import utils

from load_graph import load_reddit, load_ogb, inductive_split

def train(rank, world_size, graph, num_classes, feat_len, args, feats, labels):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', 'tcp://127.0.0.1:12347', world_size=world_size, rank=rank)

    model = SAGE(feat_len, args.num_hidden, num_classes, args.num_layers, F.relu, args.dropout).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    #opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()

    #train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    train_idx = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]
    if args.disk_feat:
        valid_idx = torch.nonzero(graph.ndata['valid_mask'], as_tuple=True)[0]
        test_idx = torch.nonzero(~(graph.ndata['train_mask'] | graph.ndata['valid_mask']), as_tuple=True)[0]
    else:
        valid_idx = torch.nonzero(graph.ndata['val_mask'], as_tuple=True)[0]
        test_idx = torch.nonzero(~(graph.ndata['train_mask'] | graph.ndata['val_mask']), as_tuple=True)[0]

    # move ids to GPU
    #train_idx = train_idx.to('cuda')
    #valid_idx = valid_idx.to('cuda')
    #test_idx = test_idx.to('cuda')
    #feats = feats.to('cuda')

    # For training, each process/GPU will get a subset of the
    # train_idx/valid_idx, and generate mini-batches indepednetly. This allows
    # the only communication neccessary in training to be the all-reduce for
    # the gradients performed by the DDP wrapper (created above).
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    #sampler = dgl.dataloading.NeighborSampler(
    #        [15, 10, 5], prefetch_node_feats=['feat'], prefetch_labels=['label'])
    #train_dataloader = dgl.dataloading.DataLoader(
    #        graph, train_idx, sampler,
    #        device='cuda', batch_size=1024, shuffle=True, drop_last=False,
    #        num_workers=0, use_ddp=True, use_uva=True)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        graph, train_idx, sampler, device='cuda', batch_size=args.batch_size,
        shuffle=True, drop_last=False, num_workers=0, use_ddp=True, use_uva=True)

    valid_dataloader = dgl.dataloading.NodeDataLoader(
        graph, valid_idx, sampler, device='cuda', batch_size=1024, 
        shuffle=True, drop_last=False, num_workers=0, use_ddp=True, use_uva=True)

    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        profiler = Profiler()
        profiler.start()
        model.train()
        t0 = time.time()
        tic_step = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            #if args.disk_feat:
            #    x1 = feats[input_nodes].cpu()
            #else:
            #    x1 = feats[input_nodes]
            #x = x1.to('cuda')
            x = feats[input_nodes].to('cuda')
            y = labels[output_nodes].to('cuda')
            if args.disk_feat:
                y = batch_labels.reshape(-1,)
            #else:
            #    x = blocks[0].srcdata['feat']
            #    y = blocks[-1].dstdata['label'][:, 0]
            blocks = [block.int().to('cuda') for block in blocks]

            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            iter_tput.append(len(output_nodes) / (time.time() - tic_step))
            if it % args.log_every == 0 and rank == 0:
                acc = MF.accuracy(y_hat, y)
                mem = torch.cuda.max_memory_allocated() / 1000000
                #print('Loss', loss.item(), 'Acc', acc.item(), 'GPU Mem', mem, 'MB')
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                      epoch, it, loss.item(), acc.item(), np.mean(iter_tput[3:]), mem))
            tic_step = time.time()

        tt = time.time()
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        epoch_time = tt - t0
        if rank == 0:
            print('Epoch Time(s): {:.4f}'.format(epoch_time))
        if epoch % args.eval_every == 0 and epoch != 0:
            #eval_acc = evaluate(model, graph, feats, labels, valid_idx, 'cuda')
            model.eval()
            ys = []
            y_hats = []
            for it, (input_nodes, output_nodes, blocks) in enumerate(valid_dataloader):
                with torch.no_grad():
                    x = feats[input_nodes].to('cuda')
                    ys.append(labels[output_nodes].to('cuda'))
                    y_hats.append(model.module(blocks, x))
            val_acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys)) / world_size
            dist.reduce(val_acc, 0)
            if rank == 0:
                print('Validation Acc {:.4f}'.format(val_acc))
        dist.barrier()

    if rank == 0 and epoch >= 5:
        avg += epoch_time
    #'''
    model.eval()
    with torch.no_grad():
        # since we do 1-layer at a time, use a very large batch size
        #pred = model.module.inference(graph, device='cuda', batch_size=2**16)
        pred = model.module.inference(graph, feats, 'cuda', batch_size=2**16)
        if rank == 0:
            test_acc = MF.accuracy(pred[test_idx], labels[test_idx])
            #print('Test acc:', acc.item())
            print('Test Acc: {:.4f}'.format(test_acc))
    #test_acc = evaluate(model, graph, feats, labels, test_idx, 'cuda')
    if rank == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--rootdir', type=str, default='../dataset/')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument('--disk-feat', action='store_true', help="Put features on disk")
    args = argparser.parse_args()

    if args.disk_feat:
        dataset_dir = args.rootdir + args.dataset
        print(dataset_dir)
        graphs, _ = dgl.load_graphs(osp.join(dataset_dir, "graph.dgl"))
        g = graphs[0]
        for k, v in g.ndata.items():
            if k.endswith('_mask'):
                g.ndata[k] = v.bool()
        feat_shape_file = osp.join(dataset_dir, "feat.shape")
        feat_file = osp.join(dataset_dir, "feat.feat")
        shape = tuple(utils.memmap(feat_shape_file, mode='r', dtype='int64', shape=(2,)))
        labels = g.ndata['label']
        node_features = utils.memmap(feat_file, random=True, mode='r', dtype='float32', shape=shape)
        n_classes = torch.max(g.ndata['label']).item() + 1
        feat_len = node_features.shape[1]
    else:
        #dataset = DglNodePropPredDataset(args.dataset)
        #g, labels = dataset[0]
        #g.ndata['label'] = labels
        g, n_classes = load_ogb(name=args.dataset, root=args.rootdir)
        g.create_formats_()     # must be called before mp.spawn().
        #n_classes = dataset.num_classes
        # use all available GPUs
        labels = g.ndata.pop('labels')
        feat_len = g.ndata['feat'].shape[1]
        #node_features = g.ndata['feat']
        node_features = g.ndata.pop('features')

    #if not args.data_cpu:
    #    labels = labels.to('cuda')
    #    node_features = node_features.to('cuda')
    #split_idx = dataset.get_idx_split()
    n_procs = torch.cuda.device_count()
    nv = g.number_of_nodes()
    ne = g.number_of_edges()
    print('|V|: {}, |E|: {}, #classes: {}, feat_length: {}, num_GPUs: {}'.format(nv, ne, n_classes, feat_len, n_procs))

    # Tested with mp.spawn and fork.  Both worked and got 4s per epoch with 4 GPUs
    # and 3.86s per epoch with 8 GPUs on p2.8x, compared to 5.2s from official examples.
    import torch.multiprocessing as mp
    mp.spawn(train, args=(n_procs, g, n_classes, feat_len, args, node_features, labels), nprocs=n_procs)
