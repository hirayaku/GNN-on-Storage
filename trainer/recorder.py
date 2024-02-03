import os, hashlib, pickle, warnings
import torch

def flatten_dict(result, out=None, prefix=''):
    if out is None:
        out = {}
    for k, v in result.items():
        if isinstance(v, dict):
            flatten_dict(v, out=out, prefix=f"{k}/")
        else:
            out[f"{prefix}{k}"] = v
    return out

class Recorder(object):
    def __init__(self, info=None):
        self.info = info
        self.log = {} # each entry in log is list[dict]
        self.md5 = hashlib.md5(str(info).encode('utf-8')).hexdigest()
        self._run = 0

    @property
    def num_runs(self):
        return len(self.log)

    def set_run(self, run):
        self._run = run
        if run not in self.log:
            self.log[run] = {}

    def __iter__(self):
        return iter(self.log)

    def add(self, iters, data: dict):
        '''
        add the data dict tagged with the iteration number into the recorder
        '''
        run_log = self.log[self._run]
        flattened = flatten_dict(data)
        for label in flattened:
            if label not in run_log:
                run_log[label] = []
        for label, dp in flattened.items():
            run_log[label].append((iters, dp))

    def get_series(self, run, *labels, as_tuple=False):
        '''
        get the data series of the given label:
        { label: [ (iters, datapoints) ], ... }
        '''
        if len(labels) == 0:
            return self.get_series(run, *self.log[run].keys())

        out = {}
        for k in labels:
            series = self.log[run][k]
            series = sorted(series, key=lambda xy: xy[0])
            if as_tuple:
                series = tuple(zip(*series))
            out[k] = series
        return out

    def get_data(self, run, *labels):
        '''
        get the data lists of the given labels:
        { label: [datapoints], ... }
        '''
        out = self.get_series(run, *labels, as_tuple=True)
        for k in out:
            out[k] = out[k][1]
        return out

    def current_acc(self):
        '''
        return the best validation accuracy so far and the correspondent test acc,
        if they exist in the current run
        '''
        run_log = self.log[self._run]
        result = {'val/acc': 0}
        if 'val/acc' in run_log:
            val_data = self.get_series(self._run, 'val/acc', as_tuple=True)['val/acc']
            val_acc = 100 * torch.tensor(val_data[1])
            val_best = val_acc.max().item()
            val_idx = val_acc.argmax().item()
            epoch = val_data[0][val_idx]
            result['val/acc'] = val_best
            result['epoch'] = epoch
        if 'test/acc' in run_log:
            test_data = self.get_series(self._run, 'test/acc', as_tuple=True)['test/acc']
            try:
                idx =  test_data[0].index(result['epoch'])
                result['test/acc'] = 100 * test_data[1][idx]
            except ValueError:
                warnings.warn(f"No test results for Epoch {result['epoch']}")
                result['test/acc'] = list(zip(*test_data))
        return result

    def stdmean(self, epochs=None):
        return stdmean_acc(self, epochs)

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        fname = f"{folder}/{self.md5}.pkl"
        i = 1
        while os.path.exists(fname):
            warnings.warn(f"trace exists for config: {str(self.info)}")
            fname = f"{folder}/{self.md5}_{i}.pkl"
            i += 1
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp)

def stdmean(logger: Recorder, *labels, summarize=None):
    '''
    compute the stdmean of logged data with the given labels across all runs
    customized by the `summarize` fn
    '''
    if summarize is None:
        summarize = lambda x: x
    series_dict = {}
    for run in logger:
        run_dict = logger.get_series(run, *labels, as_tuple=True)
        summary_dict = summarize(run_dict)
        for new_label in summary_dict:
            if new_label not in series_dict:
                series_dict[new_label] = []
            series_dict[new_label].append(summary_dict[new_label])
    stdmean_dict = {}
    for label in series_dict:
        t = torch.tensor(series_dict[label])
        stdmean_dict[label] = [t.mean().item(), t.std().item()]
    return stdmean_dict

def get_acc(val_test, epochs=None):
    val_epoch = torch.tensor(val_test['val/acc'][0])
    val_acc = 100 * torch.tensor(val_test['val/acc'][1])
    if epochs is not None:
        val_acc = val_acc[val_epoch <= epochs]
    valid = val_acc.max().item()
    epoch = val_epoch[val_acc.argmax()].item()
    idx = val_test['test/acc'][0].index(epoch)
    test = 100 * val_test['test/acc'][1][idx]
    return {'val/acc' : valid, 'test/acc': test}

def stdmean_acc(logger: Recorder, epochs=None):
    summarize = lambda acc: get_acc(acc, epochs=epochs)
    return stdmean(logger, 'val/acc', 'test/acc', summarize=summarize)

if __name__ == "__main__":
    out = flatten_dict({1 : {1 : 'a', 2: 'b'}})
    out.update({'3': 'c'})
    print(out)
