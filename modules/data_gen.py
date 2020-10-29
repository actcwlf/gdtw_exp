import numpy as np
from auxiliary.dataset import load_ucr
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pathlib as path

def transform(y: np.array, n: int):
    # print(y.shape)
    n1 = np.random.randint(int(n / 3))
    n2 = np.random.randint(n - n1)
    n3 = n - n1 - n2
    i1 = [np.random.randint(2)+1 for i in range(n1)]
    i2 = [np.random.randint(4)+1 for i in range(n2)]
    i3 = [np.random.randint(16)+1 for i in range(n3)]
    new_x = np.random.dirichlet(i1 + i2 + i3).cumsum()
    new_x = new_x / new_x.max() * (y.shape[0] - 1)
    x = np.arange(y.shape[0])
    f = interp1d(x, y)
    return f(new_x)


def gen_series(x):
    n = int(np.random.randn() * x.shape[0] / 5 + x.shape[0])
    x_t = transform(x, n)
    return x_t


def equal_interp(y, n):
    x = np.arange(y.shape[0])
    new_x = np.arange(n)
    new_x = new_x / new_x.max() * (y.shape[0] - 1)
    f = interp1d(x, y)
    return f(new_x)


def processs_data(data_name):
    X_tr, y_tr, X_te, y_te = load_ucr("../../data/ucr2015", data_name)

    # X_tr_p1 = X_tr[y_tr == 1]
    # X_te_p1 = X_te[y_te == 1]
    #
    # print(X_tr_p1.shape)
    # print(X_te_p1.shape)

    print(f"{data_name}, length: {X_tr.shape[1]} train: {X_tr.shape[0]} test:{X_te.shape[0]}")

    target = np.unique(y_tr)
    try:
        ucr_d_dir = path.Path(f"../../data/ucr2015_d/{data_name}")
        ucr_d_dir.mkdir(exist_ok=True)
    except:
        pass
    with open(f"../../data/ucr2015_d/{data_name}/{data_name}_TRAIN", "w") as f:
        for t in target:
            for x in X_tr[y_tr == t]:
                x_t = gen_series(x)
                line = ','.join(map(str, [t] + list(x_t)))
                f.write(line+'\n')


    with open(f"../../data/ucr2015_d/{data_name}/{data_name}_TEST", "w") as f:
        for t in target:
            for x in X_te[y_te == t]:
                x_t = gen_series(x)
                line = ','.join(map(str, [t] + list(x_t)))
                f.write(line+'\n')



    # x = X_tr_p1[1,:]
    # plt.plot(x)
    #
    # x2 = transform(x, 400)
    # plt.plot(x2)
    #
    # x3 = equal_interp(x2, 270)
    # plt.plot(x3)
    #
    # plt.show()

if __name__ == "__main__":
    ucr_dir = path.Path("../../data/ucr2015")
    # ctx = {"settings": SETTINGS, 'recorder': init_recorder(SETTINGS)}

    for data_name in sorted(ucr_dir.iterdir()):
        # print(data_name.name)
        # if data_name.name != 'ECGFiveDays':
        #     continue
        # if data_name.name == 'Adaic':
        processs_data(data_name.name)
        # break