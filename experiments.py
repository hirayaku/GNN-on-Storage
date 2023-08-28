import os, subprocess
import argparse, json, json5
import tempfile

DEFAULT_CONF = {
    "mode": "local", # select the running mode of this script: dry-run, local or slurm
    "env": {
        "verbose": False,
        "cwd": os.getcwd(),
        "ncpus": 32,
        "ngpus": 1,
        "cuda": 0,
        "mem": "128G",
        "trace": None,
    },
    "dataset": {
        # name of the dataset
        "name": None,
        # root path of the dataset files
        "root": "datasets",
    },
    "model": {
        "arch": "sage",
        "num_hidden": 256,
        "num_layers": 3,
        "dropout": 0.5, # no effects so far
        "edge_drop": 0.2, # no effects
        "node_drop": 0.1, # no effects
        "lr": 1e-3,
        "lr_schedule": None,
        "wt_decay": 0,

        "runs": 5,
        "epochs": 100,
        "eval_every": 1,
        "log_every": 100,
        "extra": "",
    },
    "sample": {
        "train": [ {
                "sampler": "cluster",
                "partition": "fennel-lb",
                "P": 64,
                "batch_size": 8,
                "pivots": True,
                "recycle": False,
                "num_workers": 8,
            }, {
                "sampler": "ns",
                "fanout": "15,10,5",
                "batch_size": 1000,
                "num_workers": 24,
            },
        ],
        "eval": [ {
                "sampler": "ns",
                "fanout": "20,20,20",
                "batch_size": 500,
                "num_workers": 8,
            }
        ],
    }
}

def expand_config_lists(conf: dict, **kw):
    '''
    generate a list of command flags for all specified jobs
    '''
    def tolist(value):
        if type(value) is list:
            return value
        else:
            return [value]
    env = {}
    flags = {}
    for grp, grp_dict in conf.items():
        new_dict = {**DEFAULT_CONF[grp], **grp_dict}
        if grp != "env":
            for k in new_dict:
                flags[k] = tolist(kw[k]) if k in kw else tolist(new_dict[k])
        else:
            env = { kw[k] if k in kw else new_dict[k] for k in new_dict }

    num_configs = 1
    for opt, value in flags.items():
        if len(value) > num_configs:
            num_configs = len(value)

    expanded_hparams = {}
    for opt, value in flags.items():
        assert len(value) == 1 or len(value) == num_configs, \
            f'Invalid {opt}: {value}. Need 1 or {num_configs} items'
        if len(value) == 1:
            expanded_hparams[opt] = value * num_configs
        else:
            expanded_hparams[opt] = value

    args_list = [
        {opt: expanded_hparams[opt][i] for opt in expanded_hparams}
        for i in range(num_configs)
        ]
    return env, args_list 

def merge_dict(d1: dict, d2: dict|None) -> dict:
    if d2 is None:
        return d1
    d3 = {}
    for k in d1:
        if k not in d2:
            d3[k] = d1[k]
        else:
            if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                d3[k] = merge_dict(d1[k], d2[k])
            else:
                d3[k] = d2[k]
    for k in d2:
        if k not in d1:
            d3[k] = d2[k]
    return d3

def get_config():
    parser = argparse.ArgumentParser(
        description="HierBatching experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, default='conf/arxiv.json5',
        help="specify a json configuration file")
    parser.add_argument('--mode', type=str, default='local',
        help="running mode of the script: dry-run, local, slurm")
    args = parser.parse_args()
    with open(args.config) as fp:
        config = json5.load(fp)
    config['mode'] = args.mode
    return merge_dict(DEFAULT_CONF, config)

# common prefix
SLURM_PREFIX = '''#!/bin/bash
#SBATCH --gres=gpu:{ngpu}
#SBATCH --cpus-per-task={ncpu}
#SBATCH --mem={mem}
#SBATCH --time=24:00:00
#SBATCH --output={trace}
#SBATCH --open-mode=append

source $HOME/.bashrc
conda activate gnn
cd {cwd}
export DGL_PREFETCHER_TIMEOUT={prefetch_timeout}
set -eux
'''

COMMAND = '''
mkdir -p traces/{dataset};
python3 -u trainer.py --dataset {dataset} --gpu {gpu} --root {root} --model {model} --comment "{comment}" \
--sample-L1 {sample_L1} --sample-L2 {sample_L2} \
--psize {psize} --bsize {bsize} --bsize2 {minibatch} --bsize3 {eval_minibatch} --popular-ratio {sratio} \
--num-hidden {num_hidden} --lr {lr} --lr-decay {lr_decay} --lr-step {lr_step} --part {part} --recycle {recycle} \
--n-layers {layers} --fanout {fanout} --test-fanout {eval_fanout} --n-epochs {epochs} --log-every {log_every} --eval-every {eval_every} \
--num-workers {num_workers} --runs {runs} {extra} '''

# run_ns ='''
# mkdir -p traces/{dataset};
# python3 -u trainer_ns.py --dataset {dataset} --gpu {gpu} --root {root} --model {model} --comment "{comment}" \
# --num-hidden {num_hidden} --bsize2 {minibatch} --bsize3 {eval_minibatch} --lr {lr} --lr-decay {lr_decay} --lr-step {lr_step} \
# --n-layers {layers} --fanout {fanout} --test-fanout {eval_fanout} --n-epochs {epochs} --log-every {log_every} --eval-every {eval_every} \
# --num-workers {num_workers} --runs {runs} {extra} '''
# def make_job(config, pid):
#     if config['method'] == 'HB':
#         run_cmd = run_hb 
#     elif config['method'] == 'NS':
#         run_cmd = run_ns 
#     else:
#         raise NotImplementedError(f'Method not supported: {config["method"]}')

#     return f'''({run_cmd.format(**config)}) &
# {pid}=$!
# '''

if __name__ == "__main__":
    run_config = get_config()

    with tempfile.NamedTemporaryFile(mode='w') as temp:
        json_config = json.dumps(run_config)
        temp.write(json_config)
        temp.flush()
        if run_config['mode'] == 'dry-run':
            print("In dry-run mode, commands not run")
            print("Config:")
            print(json_config)
        elif run_config['mode'] == 'local':
            subprocess.run(['python3', 'hb_acc.py', '--config', temp.name])
        else:
            ...

        # script.write(str.encode(prefix))
        # pids = []
        # for num, c in enumerate(configs):
        #     script.write(str.encode(make_job(c, f"pid{num}")))
        #     pids.append(f"pid{num}")
        #     if (num+1) % env['cjobs'] == 0 or num == len(configs)-1:
        #         wait_cmd = ['wait'] + ["${" + pid + "}" for pid in pids] + ['\n']
        #         script.write(str.encode(' '.join(wait_cmd)))
        #         pids = []
        # script.flush()
        # print("Commands written to", script.name)
