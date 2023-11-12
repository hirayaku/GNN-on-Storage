'''
This is the script to run experiments in PyG to obtain both accuracy and runtime results
'''
import os, time, random, sys, gc, logging, json5
from torch.optim import lr_scheduler
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
from trainer.helpers import get_model, get_optimizer, get_dataset
from trainer.helpers import train, eval_batch, eval_full
from trainer.recorder import Recorder

def train_with(conf: dict):
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
    dataset = NodePropPredDataset(root, mmap=(False, True), random=True, formats=('csc',))
    indices = dataset.get_idx_split()
    out_feats = dataset.num_classes
    in_feats = dataset[0].x.shape[1]
    model = get_model(in_feats, out_feats, params)
    model = model.to(device)

    sample_conf = conf['sample']
    train_conf = sample_conf['train'][0]
    train_sizes = list(map(int, train_conf['fanout'].split(',')))
    train_loader = NeighborLoader(
        dataset[0], input_nodes=indices['train'],
        num_neighbors=train_sizes,
        batch_size=train_conf['batch_size'],
        shuffle=True,
        num_workers = train_conf['num_workers'],
        persistent_workers = train_conf['num_workers'] > 0,
    )

    eval_conf = sample_conf['eval']
    val_loader, test_loader = None, None
    if eval_conf is not None:
        eval_sizes = list(map(int, eval_conf['fanout'].split(',')))
        val_loader = NeighborLoader(
            dataset[0], input_nodes=indices['valid'],
            num_neighbors=eval_sizes,
            batch_size=eval_conf['batch_size'],
            num_workers = eval_conf['num_workers'],
            persistent_workers = eval_conf['num_workers'] > 0,
        )
        test_loader = NeighborLoader(
            dataset[0], input_nodes=indices['test'],
            num_neighbors=eval_sizes,
            batch_size=eval_conf['batch_size'],
            num_workers = eval_conf['num_workers'],
            persistent_workers = eval_conf['num_workers'] > 0,
        )

    recorder = Recorder(conf)
    for run in range(params['runs']):
        main_logger.info(f"Starting Run No.{run}")
        recorder.set_run(run)
        seed_everything(run + seed)
        model.reset_parameters()
        optimizer, lr_scheduler = get_optimizer(model, params)
        main_logger.info(f"LR scheduler: {lr_scheduler}")
        gc.collect()
        torch.cuda.empty_cache()

        for e in range(params['epochs']):
            train_loss, train_acc, *train_info = train(model, optimizer, train_loader, device=device)
            mean_edges, epoch_time = train_info[2], train_info[-1]
            recorder.add(e, {'train': {'loss': train_loss, 'acc': train_acc, 'time': epoch_time}})
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
                if lr_scheduler is not None:
                    lr_scheduler.step(val_loss)

    main_logger.info(f"All runs finished with the config below: {json5.dumps(conf, indent=2)}")
    main_logger.info(f"Results: {recorder.stdmean()}")
    recorder.save(env['outdir'])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args, _ = parser.parse_known_args()
    with open(args.config) as fp:
        conf = json5.load(fp)
    main_logger.info(f"Using the config below: {json5.dumps(conf, indent=2)}")

    import torch.multiprocessing
    #  torch.multiprocessing.set_sharing_strategy('file_system')
    train_with(conf)

