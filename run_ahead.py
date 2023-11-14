import copy, time, random, sys, gc, logging, json5
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

def round_acc(acc: dict, decimal=2):
    return {
        k: round(acc[k], decimal) for k in acc
    }

def train_with(conf: dict, keep_eval=True):
    env = conf['env']
    seed = env.get('seed', random.randint(0, 1024**3))
    profile = env.get('profile', False)
    env['seed'] = seed
    if env['verbose']:
        main_logger.setLevel(logging.DEBUG)
    if 'cuda' not in env:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda:{}'.format(env['cuda']))
    main_logger.info(f"Training with GPU:{device.index}")

    dataset_conf, params = conf['dataset'], conf['model']
    dataset = NodePropPredDataset(
        dataset_conf['root'], mmap=(False, True), formats='csc'
    )
    out_feats = dataset.num_classes
    in_feats = dataset[0].x.shape[1]
    if not keep_eval:
        dataset = None
    runahead = params.get('train_runahead', 1)
    ckpt_label = params.get('ckpt', None)
    eval_test = params.get('eval_test', False)

    sample_conf = conf['sample']
    train_conf, eval_conf = sample_conf['train'], sample_conf['eval']
    assert eval_conf is not None

    recorder = Recorder(conf)
    for run in range(params['runs']):
        main_logger.info(f"Starting Run No.{run}")
        recorder.set_run(run)
        seed_everything(run + seed)
        model = get_model(in_feats, out_feats, params)
        model = model.to(device)
        optimizer, lr_scheduler = get_optimizer(model, params)
        main_logger.info(f"LR scheduler: {lr_scheduler}")
        gc.collect(); torch.cuda.empty_cache()

        mp.set_sharing_strategy('file_system')
        train_loader = HierarchicalDataLoader(dataset_conf, 'train', train_conf)

        model_ckpts = {}
        for e in range(params['epochs']):
            gc.collect() # make sure the dataloader memory gets reclaimed
            if profile:
                train_loss, train_acc, *train_info = train_profile(model, optimizer, lr_scheduler, train_loader, device=device)
            else:
                train_loss, train_acc, *train_info = train(model, optimizer, lr_scheduler, train_loader, device=device)
            mean_edges, epoch_time = train_info[2], train_info[-1]
            recorder.add(e, {'train': {'loss': train_loss, 'acc': train_acc, 'time': epoch_time}})
            main_logger.info(
                f"Epoch {e:3d} | Train {train_acc*100:.2f} | Loss {train_loss:.2f} | MFG {mean_edges:.2f}"
            )
            model_ckpts[e] = copy.deepcopy(model)
            if (e + 1) % runahead != 0:
                continue
            else:
                train_loader.shutdown()

                # evaluate on the saved model_ckpts
                mp.set_sharing_strategy('file_descriptor')
                eval_dataset = NodePropPredDataset(
                    dataset_conf['root'], mmap=(False, True), random=True, formats='csc'
                )
                val_loader = NodeDataLoader(eval_dataset, 'valid', eval_conf)
                if eval_test: test_loader = NodeDataLoader(eval_dataset, 'test', eval_conf)
                for eval_epoch in sorted(model_ckpts.keys()):
                    if (eval_epoch + 1) % params.get('eval_every', 1) != 0:
                        continue
                    eval_model = model_ckpts[eval_epoch]
                    prev_best = recorder.current_acc()['val/acc']
                    with utils.parallelism(factor=8): # overcommit threads
                        val_loss, val_acc, *_ = eval_batch(
                            eval_model, val_loader, device=device, description='valid'
                        )
                        recorder.add(iters=eval_epoch, data={'val': { 'loss': val_loss, 'acc': val_acc, }})
                        if eval_test:
                            test_loss, test_acc, *_ = eval_batch(
                                eval_model, test_loader, device=device, description='test'
                            )
                            recorder.add(iters=eval_epoch, data={'test': { 'loss': test_loss, 'acc': test_acc, }})
                    curr_best = round_acc(recorder.current_acc())
                    if eval_test: main_logger.info(
                            f"Current Val: {val_acc*100:.2f} | Test: {test_acc*100:.2f} | {curr_best}"
                        )
                    else: main_logger.info(f"Current Val: {val_acc*100:.2f} | {curr_best}")
                    # checkpoint the best model so far
                    if curr_best['epoch'] == e:
                        torch.save(
                            {'run': run, 'epoch': eval_epoch, 'model': eval_model.state_dict()},
                            f'models/ckpt/{dataset_conf["name"]}-{params["arch"]}.{ckpt_label}.{run}.pt'
                        )
                    if lr_scheduler is not None:
                        lr_scheduler.step(val_loss)
                val_loader.shutdown(); val_loader = None
                if eval_test: test_loader.shutdown(); test_loader = None
                eval_dataset = None
                model_ckpts = {}

            mp.set_sharing_strategy('file_system')
            train_loader = HierarchicalDataLoader(dataset_conf, 'train', train_conf)

        train_loader.shutdown()
        if not eval_test:
            mp.set_sharing_strategy('file_descriptor')
            ckpt = torch.load(f'models/ckpt/{dataset_conf["name"]}-{params["arch"]}.{ckpt_label}.{run}.pt')
            model.load_state_dict(ckpt['model'])
            eval_dataset = NodePropPredDataset(
                dataset_conf['root'], mmap=(False, True), random=True, formats='csc'
            )
            test_loader = NodeDataLoader(eval_dataset, 'test', eval_conf)
            test_loss, test_acc, *_ = eval_batch(
                model, test_loader, device=device, description='test'
            )
            recorder.add(iters=ckpt['epoch'], data={'test': { 'loss': test_loss, 'acc': test_acc, }})

        main_logger.info(f"{round_acc(recorder.current_acc())}")

    main_logger.info(f"All runs finished with the config below: {json5.dumps(conf, indent=2)}")
    main_logger.info(f"Results: {recorder.stdmean()}")
    return recorder

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='conf/papers-hb.json5')
    parser.add_argument("--keep-eval", action="store_true",
                        help="keep evaluation dataloaders in memory")
    args, _ = parser.parse_known_args()
    with open(args.config) as fp:
        conf = json5.load(fp)
    main_logger.info(f"Using the config below: {json5.dumps(conf, indent=2)}")

    import torch.multiprocessing as mp
    recorder = train_with(conf, keep_eval=args.keep_eval)
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
