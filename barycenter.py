# Author: Mathieu Blondel
# License: Simplified BSD
import pathlib as path
import numpy as np
from tslearn.metrics import dtw, dtw_path
from tqdm import tqdm
from modules.barycenter import sdtw_barycenter
from modules.barycenter import gfsdtw_barycenter
from auxiliary.dataset import load_ucr
import time
import fsdtw
import itertools

TIMESTAMP = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))


def exp_fun(ctx, name):
    print(f"calculating {name}")
    X_tr, y_tr, X_te, y_te = load_ucr("data/ucr2015", name)

    # PATCH only for original sdtw implementation
    X_tr = X_tr.reshape(*X_tr.shape, 1)
    # END PATCH
    result = []
    for seed in tqdm(range(10), disable=True):
        r = exp_1seed(ctx, X_tr, y_tr, seed)
        result.append(r)

    result = np.array(result)
    return name, result.mean(axis=0), result.std(axis=0)


def exp_1seed(ctx, X_tr, y_tr, seed=0):
    settings = ctx['settings']
    n = 10
    # Pick n time series at random from the same class.
    rng = np.random.RandomState(seed)
    classes = np.unique(y_tr)
    k = rng.randint(len(classes))
    X = X_tr[y_tr == classes[k]]
    X = X[rng.permutation(len(X))[:n]]

    barycenter_init = sum(X) / len(X)
    result = []

    for r, gamma, q in settings['params']: # gamma, q in zip((1, 0.1, 0.01, 0.001, 0.0001, 0.00001), (20, 50, 100, 200, 500, 1000)):
        # gamma = settings['gamma']
        # q = settings['q']
        print(f"seed: {seed}, r {r}, gamma: {gamma}, q {q}")
        dtw_score = 0
        Z = None
        if settings['method'] == "softdtw":
            Z = sdtw_barycenter(X, barycenter_init, gamma=gamma, max_iter=settings['max_iter'])
        elif settings['method'] == "gfsdtw":
            Z = gfsdtw_barycenter(X, barycenter_init, gamma=gamma, q=q, radius=r, max_iter=settings['max_iter'])
        else:
            raise Exception(f'metohd `{settings["method"]}` not found')
        for x in X:
            dtw_score += (dtw(x.squeeze(), Z))**2
        result.append(dtw_score / len(X))
        # print('finish one', time.strftime('%H:%M:%S', time.localtime(time.time())))
    return np.array(result)




def get_params():
    r = [1]
    gamma = [0.05, 0.1, 0.2, 0.5, 1]
    q = [10, 20, 300, 400, 600, 1000]
    # r = [1]
    # gamma = [0.1]
    # q = [100]
    params = list(itertools.product(r, gamma, q))
    return params


SETTINGS = {
    "method": "gfsdtw", # "fsdtw"
    "version": fsdtw.__version__,
    "max_iter": 100,
    "params": get_params()
}

if __name__ == "__main__":
    ucr_dir = path.Path("data/ucr2015")
    ctx = {"settings": SETTINGS}

    for data_name in sorted(ucr_dir.iterdir()):
        print(data_name.name)
        name, r_mean, r_std = exp_fun(ctx, data_name.name)
        r = []
        for i in range(r_mean.shape[0]):
            r.append(r_mean[i])
            r.append(r_std[i])
        print(name, *r)
        break



