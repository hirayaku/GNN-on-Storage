import os, time, random, sys, gc, logging, json5
main_logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d[%(levelname)s] %(module)s: %(message)s",
    datefmt='%0y-%0m-%0d %0H:%0M:%0S')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
main_logger.addHandler(stream_handler)
main_logger.setLevel(logging.INFO)

import torch
from torch_geometric import seed_everything
from torch_geometric.loader import NeighborLoader
from data.graphloader import NodePropPredDataset
from trainer.helpers import get_model, get_optimizer, get_dataset
from trainer.helpers import train, train_profile, eval_batch, eval_full
from trainer.dataloader import NodeDataLoader, PartitionDataLoader, HierarchicalDataLoader
from trainer.recorder import Recorder
import utils

def train_with(conf: dict, keep_eval=True, profile=False):
    env = conf['env']
    seed = env.get('seed', random.randint(0, 1024**3))
    env['seed'] = seed
    if env['verbose']:
        main_logger.setLevel(logging.DEBUG)
    if 'cuda' not in env:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda:{}'.format(env['cuda']))
    main_logger.info(f"Training with GPU:{device.index}")

    dataset_conf, params = conf['dataset'], conf['model']
    dataset = get_dataset(dataset_conf['root'])
    out_feats = dataset.num_classes
    in_feats = dataset[0].x.shape[1]
    del dataset
    model = get_model(in_feats, out_feats, params)
    model = model.to(device)

    sample_conf = conf['sample']
    train_conf, eval_conf = sample_conf['train'], sample_conf['eval']
    if len(train_conf) > 1:
        main_logger.info("Using the hierarchical DataLoader")
        train_loader = HierarchicalDataLoader(dataset_conf, 'train', train_conf)
    else:
        train_conf = train_conf[0]
        if train_conf['sampler'] == 'cluster':
            main_logger.info("Using the partition DataLoader")
            train_loader = PartitionDataLoader(dataset_conf, 'train', train_conf)
        else:
            main_logger.info("Using the conventional DataLoader")
            train_loader = NodeDataLoader(dataset_conf, 'train', train_conf)
    eval_full_batch = eval_conf is None
    val_loader = test_loader = None
    if not eval_full_batch and keep_eval:
        val_loader = NodeDataLoader(dataset_conf, 'valid', eval_conf)
        test_loader = NodeDataLoader(dataset_conf, 'test', eval_conf)

    recorder = Recorder(conf)
    for run in range(params['runs']):
        main_logger.info(f"Starting Run No.{run}")
        recorder.set_run(run)
        seed_everything(run + seed)
        model.reset_parameters()
        optimizer, lr_scheduler = get_optimizer(model, params)
        main_logger.info(f"LR scheduler: {lr_scheduler}")
        gc.collect(); torch.cuda.empty_cache()

        for e in range(params['epochs']):
            gc.collect() # make sure the dataloader memory gets reclaimed
            if profile:
                train_loss, train_acc, *train_info = train_profile(model, optimizer, train_loader, device=device)
            else:
                train_loss, train_acc, *train_info = train(model, optimizer, train_loader, device=device)
            mean_edges, epoch_time = train_info[2], train_info[-1]
            recorder.add(e, {'train': {'loss': train_loss, 'acc': train_acc, 'time': epoch_time}})
            main_logger.info(
                f"Epoch {e:3d} | Train {train_acc*100:.2f} | Loss {train_loss:.2f} | MFG {mean_edges:.2f}"
            )

            if e >= params.get('eval_after', 0) and (e + 1) % params['eval_every'] == 0:
                if eval_full_batch:
                    dataset = get_dataset(dataset_conf['root'])
                    indices = dataset.get_idx_split()
                    results = eval_full(model, dataset[0], device=device,
                                        masks=(indices['valid'], indices['test']))
                    val_loss, val_acc = results[0]
                    test_loss, test_acc = results[1]
                    del dataset
                elif keep_eval:
                    val_loss, val_acc, *_ = eval_batch(
                        model, val_loader, device=device, description='validation'
                    )
                    test_loss, test_acc, *_ = eval_batch(
                        model, test_loader, device=device, description='test'
                    )
                else:
                    dataset = NodePropPredDataset(
                        dataset_conf['root'], mmap=(False, True), random=True, formats='csc'
                    )
                    eval_sizes = list(map(int, eval_conf['fanout'].split(',')))
                    eval_loader = NeighborLoader(
                        dataset[0], input_nodes=dataset.get_idx_split('valid'),
                        num_neighbors=eval_sizes,
                        batch_size=eval_conf['batch_size'],
                        num_workers=eval_conf['num_workers'],
                    )
                    with utils.parallelism(factor=4): # overcommit threads
                        val_loss, val_acc, *_ = eval_batch(
                            model, eval_loader, device=device, description='validation'
                        )
                    eval_loader = NeighborLoader(
                        dataset[0], input_nodes=dataset.get_idx_split('test'),
                        num_neighbors=eval_sizes,
                        batch_size=eval_conf['batch_size'],
                        num_workers=eval_conf['num_workers'],
                    )
                    with utils.parallelism(factor=4): # overcommit threads
                        test_loss, test_acc, *_ = eval_batch(
                            model, eval_loader, device=device, description='test'
                        )
                    del eval_loader, dataset

                recorder.add(iters=e, data={
                    'val':    { 'loss': val_loss, 'acc': val_acc, },
                    'test':     { 'loss': test_loss, 'acc': test_acc, },
                })
                best_acc = recorder.current_acc()
                main_logger.info(
                    f"Current Val: {val_acc*100:.2f} | Current Test: {test_acc*100:.2f} | "
                    f"Best Val: {best_acc['val/acc']:.2f} | Test {best_acc['test/acc']:.2f}"
                )
                if lr_scheduler is not None:
                    lr_scheduler.step(val_loss)

    main_logger.info(f"All runs finished with the config below: {json5.dumps(conf, indent=2)}")
    main_logger.info(f"Results: {recorder.stdmean()}")
    return recorder

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='conf/products-ns.json5')
    parser.add_argument("--keep-eval", action="store_true",
                        help="keep evaluation dataloaders in memory")
    parser.add_argument("--profile", action="store_true", help="enable profiling")
    parser.add_argument("-v", "--verbose", action="store_true", help="display debugging messages")
    args, _ = parser.parse_known_args()
    if args.verbose:
        main_logger.setLevel(logging.DEBUG)
    with open(args.config) as fp:
        conf = json5.load(fp)
    main_logger.info(f"Using the config below: {json5.dumps(conf, indent=2)}")

    import torch.multiprocessing as mp
    # NOTE: don't change sharing strategy for baseline mmap
    mp.set_sharing_strategy('file_system')
    recorder = train_with(conf, keep_eval=args.keep_eval, profile=args.profile)
    env = conf.get('env', dict())
    if 'outdir' in env:
        recorder.save(env['outdir'])

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
