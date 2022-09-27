import os, subprocess
import argparse

parser = argparse.ArgumentParser(description="HierBatching experiment runner")
parser.add_argument('--dataset', type=str)
parser.add_argument('--root', type=str, default=f'{os.environ["DATASETS"]}/gnnos')
parser.add_argument('--method', type=str, nargs='+', default=['HB'], help='Choose between NS, HB')
parser.add_argument('--comment', type=str, default='')
parser.add_argument('--runs', type=int, default=1)
# common hparams
parser.add_argument('--model', type=str, nargs='+', default=['sage'], help='GNN models')
parser.add_argument('--num-hidden', type=int, nargs='+', default=[256], help='hidden dimensions')
parser.add_argument('--minibatch', type=int, nargs='+', default=[1024])
parser.add_argument('--eval-minibatch', type=int, nargs='+', default=[512])
parser.add_argument('--dropout', type=float, nargs='+', default=[0.5])
parser.add_argument('--lr', type=float, nargs='+', default=[1e-3])
parser.add_argument('--lr-decay', type=float, nargs='+', default=[0.9999])
parser.add_argument('--lr-step', type=int, nargs='+', default=[1000])
parser.add_argument('--wt-decay', type=float, nargs='+', default=[0])
parser.add_argument('--fanout', type=str, nargs='+', default=['15,10,5'])
# HB-related hparams below
parser.add_argument('--part', type=str, nargs='+', default=['metis'], help='partitioners')
parser.add_argument('--psize', type=int, nargs='+', default=[1024], help='number of partitions')
parser.add_argument('--bsize', type=int, nargs='+', default=[0], help='partitions in a mega-batch')
parser.add_argument('--sratio', type=float, nargs='+', default=[0.01], help='s-cache node ratio')
parser.add_argument('--recycle', type=int, nargs='+', default=[1], help='initial reuse factor')
parser.add_argument('--rho', type=float, nargs='+', default=[1.0], help='reuse multiplication factor')

args = vars(parser.parse_args())
non_hparams = ('dataset', 'root', 'comment', 'runs')
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

sbatch_hb = '''
DATASET="{dataset}" ROOT="{root}" COMMENT="{comment}" RUNS={runs} \
MODEL="{model}" HIDDEN={num_hidden} BSIZE2={minibatch} BSIZE3={eval_minibatch} LR={lr} \
FANOUT="{fanout}" LRD={lr_decay} LRS={lr_step} WD={wt_decay} DPOUT={dropout} \
PART={part} PSIZE={psize} BSIZE={bsize} PRATIO={sratio} REC={recycle} RHO={rho} \
sbatch -J {dataset}-{model}-{part}/{psize}/{bsize}/{minibatch}/{sratio} --export=ALL sbatch/{dataset}.sbatch
'''

sbatch_ns = '''
DATASET="{dataset}" ROOT="{root}" COMMENT="{comment}" RUNS={runs} \
MODEL="{model}" HIDDEN={num_hidden} BSIZE2={minibatch} BSIZE3={eval_minibatch} LR={lr} \
FANOUT="{fanout}" LRD={lr_decay} LRS={lr_step} WD={wt_decay} DPOUT={dropout} \
PART={part} PSIZE={psize} BSIZE={bsize} PRATIO={sratio} REC={recycle} RHO={rho} \
sbatch -J {dataset}-{model}-NS/{minibatch} --export=ALL sbatch/{dataset}-sg.sbatch
'''

for c in configs:
    if c['method'] == 'HB':
        print('{dataset}, {model}, {part}-{psize}/{bsize}, {sratio}'.format(**c))
        subprocess.run(['bash', '-c', sbatch_hb.format(**c)])
    elif c['method'] == 'NS':
        print('{dataset}, {model}, NS/{minibatch}'.format(**c))
        subprocess.run(['bash', '-c', sbatch_ns.format(**c)])

