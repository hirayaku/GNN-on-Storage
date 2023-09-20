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

    def get_series(self, run, *labels):
        '''
        get the data series of the given label:
        { label: [ (iters, datapoints) ], ... }
        '''
        if len(labels) == 0:
            return self.get_series(run, *self.log[run].keys())

        out = {}
        for k in labels:
            series = self.log[run][k]
            out[k] = sorted(series, key=lambda xy: xy[0])
        return out

    def get_data(self, run, *labels):
        '''
        get the data lists of the given labels:
        { label: [datapoints], ... }
        '''
        out = self.get_series(run, *labels)
        for k in out:
            out[k] = list(zip(*out[k]))[1]
        return out

    def current_acc(self):
        '''
        return the best validation accuracy so far and the correspondent test acc,
        if they exist in the current run
        '''
        run_log = self.log[self._run]
        if 'val/acc' in run_log and 'test/acc' in run_log:
            val_test = self.get_data(self._run, 'val/acc', 'test/acc')
            val_acc = 100 * torch.tensor(val_test['val/acc'])
            valid = val_acc.max().item()
            test = 100 * val_test['test/acc'][val_acc.argmax()]
            return {'val/acc' : valid, 'test/acc': test}
        else:
            return {}

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
        run_dict = logger.get_data(run, *labels)
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
    val_acc = 100 * torch.tensor(val_test['val/acc'])
    if epochs is not None:
        val_acc = val_acc[:epochs]
    valid = val_acc.max().item()
    test = 100 * val_test['test/acc'][val_acc.argmax()]
    return {'val/acc' : valid, 'test/acc': test}

def stdmean_acc(logger: Recorder, epochs=None):
    summarize = lambda acc: get_acc(acc, epochs=epochs)
    return stdmean(logger, 'val/acc', 'test/acc', summarize=summarize)

if __name__ == "__main__":
    out = flatten_dict({1 : {1 : 'a', 2: 'b'}})
    out.update({'3': 'c'})
    print(out)
