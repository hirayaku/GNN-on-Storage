import os, time, random, sys, logging

main_logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d[%(levelname)s] %(module)s: %(message)s",
    datefmt='%0y-%0m-%0d %0H:%0M:%0S')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
main_logger.addHandler(stream_handler)

import numpy as np
import torch
from trainer.helpers import get_config, get_model, get_dataset
from trainer.helpers import train, eval_batch, eval_full, train_partitioner
from trainer.dataloader import NodeDataLoader, HierarchicalDataLoader, PartitionDataLoader
from trainer.recorder import Recorder

def train_with(conf: dict):
    env = conf['env']
    if env['verbose']:
        main_logger.setLevel(logging.DEBUG)
    else:
        main_logger.setLevel(logging.INFO)
    if 'cuda' not in env:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda:{}'.format(env['cuda']))
    main_logger.info(f"Training with GPU:{device.index}")
    
    dataset_conf, params = conf['dataset'], conf['model']
    dataset = get_dataset(dataset_conf['root'])
    indices = dataset.get_idx_split()
    out_feats = dataset.num_classes
    in_feats = dataset[0].x.shape[1]
    model = get_model(in_feats, out_feats, params)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    if params.get('lr_schedule', None) == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=params['lr_decay'],
            patience=params['lr_step'],
        )
    else:
        lr_scheduler = None
    main_logger.info(f"LR scheduler: {lr_scheduler}")

    sample_conf = conf['sample']
    train_loader = PartitionDataLoader(dataset_conf, env, 'train', sample_conf['train'][0])
    # if len(sample_conf['train']) == 2:
    #     train_loader = HierarchicalDataLoader(dataset_conf, env, 'train', sample_conf['train'])
    # else:
    #     train_loader = NodeDataLoader(dataset_conf, env, 'train', sample_conf['train'][0])
    if sample_conf['eval'] is not None:
        val_loader = NodeDataLoader(dataset_conf, env, 'valid', sample_conf['eval'])
        test_loader = NodeDataLoader(dataset_conf, env, 'test', sample_conf['eval'])

    recorder = Recorder(conf)
    recorder.set_run(0)
    for e in range(params['epochs']):
        # train_loss, train_acc, *train_info = train(model, optimizer, train_loader, device=device)
        train_loss, train_acc, *train_info = train_partitioner(model, optimizer, train_loader, device=device)
        mean_edges = train_info[2]
        recorder.add(e, {'train': {'loss': train_loss, 'acc': train_acc}})
        main_logger.info(
            f"Epoch {e:3d} | Train {train_acc*100:.2f} | Loss {train_loss:.2f} | MFG {mean_edges:.2f}"
        )

        if (e + 1) % params['eval_every'] == 0:
            if sample_conf['eval'] is None:
                results = eval_full(model, dataset[0], device=device,
                                    masks=(indices['valid'], indices['test']))
                val_loss, val_acc = results[0]
                test_loss, test_acc = results[1]
            else:
                val_loss, val_acc, *_ = eval_batch(model, val_loader, device=device,
                                                description='validation')
                test_loss, test_acc, *_ = eval_batch(model, test_loader, device=device,
                                                    description='test')
            recorder.add(iters=e, data={
                'val':    { 'loss': val_loss, 'acc': val_acc, },
                'test':     { 'loss': test_loss, 'acc': test_acc, },
            })
            best_acc = recorder.current_acc()
            main_logger.info(
                f"Current Val: {val_acc*100:.2f} | Current Test: {test_acc*100:.2f} | "
                f"Best Val: {best_acc['val/acc']:.2f} | Test {best_acc['test/acc']:.2f}"
            )
    best_acc = recorder.current_acc()
    main_logger.info(
        f"Current run finished with the following config:\n{conf}\n"
        f"Best Val: {best_acc['val/acc']:.2f}. Final Test: {best_acc['test/acc']:.2f}"
    )

if __name__ == '__main__':
    # seed = random.randint(0,1024**3)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    # print(f"\n## [{os.environ['HOSTNAME']}] {args.comment} seed:{seed}")
    # print(time_stamp)
    # print(args)
    # torch.cuda.set_device(args.gpu)
    # device = torch.device(f'cuda:{torch.cuda.current_device()}')
    # print(f"Training with GPU: {device}")

    conf = get_config()
    train_with(conf)

    # print(f"""----Data statistics------
    # #Nodes {n_nodes}
    # #Edges {n_edges}
    # #Classes/Labels (multi binary labels) {n_classes}
    # #Train samples {n_train_samples}
    # #Val samples {n_val_samples}
    # #Test samples {n_test_samples}
    # #Labels     {g.ndata['label'].shape}
    # #Features   {g.ndata['feat'].shape}"""
    # )

    # log_path = f"log/{args.dataset}/HB-{model}-{partition}-c{args.popular_ratio}" \
    #         + f"-r{args.recycle}*{args.rho}-p{args.psize}-b{args.bsize}-{time_stamp}"
    # tb_writer = SummaryWriter(log_path, flush_secs=5)
    # try:
    #     accu = train(args, data, partitioner, tb_writer)
    # except:
    #     shutil.rmtree(log_path)
    #     print("** removed tensorboard log dir **")
    #     raise
    # tb_writer.add_hparams({
    #     'seed': seed,'model': model,'num_hidden': args.num_hidden, 'fanout': str(args.fanout),
    #     'use_incep': args.use_incep, 'mlp': args.mlp, 'lr': args.lr, 'lr-decay': args.lr_decay,
    #     'dropout': args.dropout, 'weight-decay': args.wt_decay,
    #     'partition': args.part, 'psize': args.psize, 'bsize': args.bsize, 'bsize2': args.bsize2,
    #     'rho': args.rho, 'recycle': args.recycle, 'popular_ratio': args.popular_ratio,
    #     },
    #     {'hparam/val_acc': accu[0].item(), 'hparam/test_acc': accu[1].item() }
    #     )
    # tb_writer.close()
