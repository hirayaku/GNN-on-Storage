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
from trainer.helpers import get_model
from trainer.helpers import eval_batch, eval_full
from trainer.dataloader import NodeDataLoader
from trainer.recorder import Recorder
import utils

def round_acc(acc: dict, decimal=2):
    return {
        k: round(acc[k], decimal) for k in acc
    }

def eval_with(conf: dict, run: int = 0):
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
        dataset_conf['root'], mmap=(False, True), random=True, formats='csc'
    )
    dataset[0].share_memory_()
    out_feats = dataset.num_classes
    in_feats = dataset[0].x.shape[1]
    eval_test = params.get('eval_test', False)
    model_ckpt = params.get('ckpt', None)
    model = get_model(in_feats, out_feats, params)
    ckpt = torch.load(f'models/ckpt/{dataset_conf["name"]}-{params["arch"]}.{model_ckpt}.{run}.pt')
    model.load_state_dict(ckpt['model'])
    model = model.to(device)

    eval_conf = conf['sample']['eval']
    eval_full_batch = eval_conf is None
    if eval_full_batch:
        indices = dataset.get_idx_split()
        results = eval_full(model, dataset[0], device=device,
                            masks=(indices['valid'], indices['test']))
        val_loss, val_acc = results[0]
        test_loss, test_acc = results[1]
    else:
        dataloader = NodeDataLoader(dataset, 'valid', eval_conf)
        with utils.parallelism(factor=8): # overcommit threads
            val_loss, val_acc, *_ = eval_batch(
                model, dataloader, device=device, description='valid'
            )
        dataloader = NodeDataLoader(dataset, 'test', eval_conf)
        with utils.parallelism(factor=8): # overcommit threads
            test_loss, test_acc, *_ = eval_batch(
                model, dataloader, device=device, description='valid'
            )

    return val_acc, test_acc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='conf/papers-hb.json5')
    parser.add_argument("-r", "--run", type=int, default=0)
    args, _ = parser.parse_known_args()
    with open(args.config) as fp:
        conf = json5.load(fp)
    main_logger.info(f"Using the config below: {json5.dumps(conf, indent=2)}")
    val_acc, test_acc = eval_with(conf, run=args.run)
    main_logger.info(f"Final Val: {val_acc*100:.2f}, Test: {test_acc*100:.2f}")

