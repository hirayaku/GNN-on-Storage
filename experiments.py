import os, subprocess
import argparse

parser = argparse.ArgumentParser(description="HierBatching experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--root', type=str, default=f'{os.environ["DATASETS"]}/gnnos')
parser.add_argument('--comment', type=str, default='')
parser.add_argument('--mem', type=str, default='256G', help='request memory budget')
parser.add_argument('--dataset', type=str, help='dataset')
parser.add_argument('--prefetch-timeout', type=int, default=60, help='prefetch timeout for DGL dataloaders')
# common hparams
parser.add_argument('--runs', type=int, nargs='+', default=[1], help='')
parser.add_argument('--method', type=str, nargs='+', default=['HB'], help='Choose between NS, HB')
parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='GPU device')
parser.add_argument('--epochs', type=int, nargs='+', default=[100], help='number of epochs')
parser.add_argument('--model', type=str, nargs='+', default=['sage'], help='GNN models')
parser.add_argument('--num-hidden', type=int, nargs='+', default=[256], help='hidden dimensions')
parser.add_argument('--minibatch', type=int, nargs='+', default=[1024], help='minibatch size')
parser.add_argument('--eval-minibatch', type=int, nargs='+', default=[512], help='')
parser.add_argument('--eval-every', type=int, nargs='+', default=[2], help='epochs to eval')
parser.add_argument('--dropout', type=float, nargs='+', default=[0.5], help='dropout')
parser.add_argument('--lr', type=float, nargs='+', default=[1e-3], help='lr')
parser.add_argument('--lr-decay', type=float, nargs='+', default=[0.9999], help='')
parser.add_argument('--lr-step', type=int, nargs='+', default=[100000], help='')
parser.add_argument('--wt-decay', type=float, nargs='+', default=[0], help='')
parser.add_argument('--fanout', type=str, nargs='+', default=['15,10,5'], help='training fanout')
parser.add_argument('--test-fanout', type=str, nargs='+', default=['20,20,20'], help='eval fanout')
parser.add_argument('--log-every', type=int, nargs='+', default=[100], help='')
parser.add_argument('--num-workers', type=int, nargs='+', default=[4], help='')
# HB-related hparams below
parser.add_argument('--part', type=str, nargs='+', default=['metis'], help='partitioners')
parser.add_argument('--psize', type=int, nargs='+', default=[1024], help='number of partitions')
parser.add_argument('--bsize', type=int, nargs='+', default=[0], help='partitions in a mega-batch')
parser.add_argument('--sratio', type=float, nargs='+', default=[0.01], help='s-cache node ratio')
parser.add_argument('--recycle', type=float, nargs='+', default=[1], help='initial reuse factor')
parser.add_argument('--rho', type=float, nargs='+', default=[1.0], help='reuse multiplication factor')

args = vars(parser.parse_args())
non_hparams = ('root', 'comment', 'mem', 'dataset', 'prefetch_timeout')
def gen_config(args: dict):
    num_configs = 1
    for opt, value in args.items():
        if opt not in non_hparams:
            if len(value) > num_configs:
                num_configs = len(value)

    full_args = {}
    for opt, value in args.items():
        if opt in non_hparams:
            full_args[opt] = [value] * num_configs
        else:
            assert len(value) == 1 or len(value) == num_configs, \
                f'Invalid {opt}: {value}. Need 1 or {num_configs} items'
            if len(value) == 1:
                full_args[opt] = value * num_configs
            else:
                full_args[opt] = value

    configs_list = [{opt: full_args[opt][i] for opt in full_args} for i in range(num_configs)]
    return configs_list

configs = gen_config(args)

gpu_set = set(args['gpu'])

# common prefix of sbatch scripts
prefix = f'''#!/bin/bash

#SBATCH --gres=gpu:{len(gpu_set)}
#SBATCH --cpus-per-task=40
#SBATCH --mem={args['mem']}
#SBATCH --time=24:00:00
#SBATCH --output=traces/{args['dataset']}.log
#SBATCH --open-mode=append

source $HOME/.bashrc
conda activate gnn
cd $HOME/proj/GNNoS
mkdir -p traces/{args['dataset']}
date
export DGL_PREFETCHER_TIMEOUT={args['prefetch_timeout']}

set -eu
'''

#  srun --output=traces/{dataset}/{method}.{model}.c{sratio}.r{recycle}.t{bsize}.p{psize}.b{minibatch}.out --open-mode=append \
sbatch_hb = '''
    python3 -u trainer_hb.py --dataset {dataset} --gpu {gpu} --root {root} --model {model} --comment "{comment}" \
    --psize {psize} --bsize {bsize} --bsize2 {minibatch} --bsize3 {eval_minibatch} --popular-ratio {sratio} \
    --num-hidden {num_hidden} --lr {lr} --lr-decay {lr_decay} --lr-step {lr_step} --part {part} --recycle {recycle} \
    --fanout 15,10,5 --test-fanout {test_fanout} --n-epochs {epochs} --log-every {log_every} --eval-every {eval_every} \
    --num-workers {num_workers} --runs {runs} \
    >> traces/{dataset}/{method}.{model}.c{sratio}.r{recycle}.t{bsize}.p{psize}.b{minibatch}.out 2>&1
'''
sbatch_ns ='''
    python3 -u trainer_ns.py --dataset {dataset} --gpu {gpu} --root {root} --model {model} --comment "{comment}" \
    --num-hidden {num_hidden} --bsize2 {minibatch} --bsize3 {eval_minibatch} --lr {lr} --lr-decay {lr_decay} --lr-step {lr_step} \
    --fanout 15,10,5 --test-fanout {test_fanout} --n-epochs {epochs} --log-every {log_every} --eval-every {eval_every} \
    --num-workers {num_workers} --runs {runs} \
    >> traces/{dataset}/{method}.{model}.b{minibatch}.out 2>&1
'''

def make_parallel(config, pid):
    if config['method'] == 'HB':
        sbatch_cmd = sbatch_hb
    elif config['method'] == 'NS':
        sbatch_cmd = sbatch_ns
    else:
        raise NotImplementedError(f'Method not supported: {config["method"]}')

    return f'''({sbatch_cmd.format(**config)}) &
{pid}=$!
'''

#  for c in configs:
#      if c['method'] == 'HB':
#          print('{dataset}, {model}, {part}-{psize}/{bsize}, {sratio}'.format(**c))
#          subprocess.run(['bash', '-c', sbatch_hb.format(**c)])
#      elif c['method'] == 'NS':
#          print('{dataset}, {model}, NS/{minibatch}'.format(**c))
#          subprocess.run(['bash', '-c', sbatch_ns.format(**c)])
#

import tempfile

with tempfile.NamedTemporaryFile(delete=False) as sbatch_file:
    sbatch_file.write(str.encode(prefix))
    for num, c in enumerate(configs):
        sbatch_file.write(str.encode(make_parallel(c, f"pid{num}")))
    wait_cmd = ['wait'] + ["${" + f"pid{num}" + "}" for num in range(len(configs))] + ['\n']
    sbatch_file.write(str.encode(' '.join(wait_cmd)))
    print("Commands written to", sbatch_file.name)

subprocess.run(['bash', '-c', f'sbatch -J {args["comment"]}:{sbatch_file.name} {sbatch_file.name}'])

