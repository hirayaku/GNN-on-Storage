from trainer.dataloader import PartitionDataLoader, NodeDataLoader
from trainer.helpers import get_config, get_model, get_dataset, train

import os, time, random, sys, logging, torch

main_logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d[%(levelname)s] %(module)s: %(message)s",
    datefmt='%0y-%0m-%0d %0H:%0M:%0S')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
main_logger.addHandler(stream_handler)


conf = get_config()

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
# train_loader = NodeDataLoader(dataset_conf, env, 'train', sample_conf['train'][0])

print(train_loader)

train_loss, train_acc, *train_info = train(model, optimizer, train_loader, device=device)


