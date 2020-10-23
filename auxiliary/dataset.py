import pandas as pd
import numpy as np
import os
import pathlib as path


def load(p: path.Path):
    raw_data = np.loadtxt(str(p.absolute()),delimiter=',')
    label = raw_data[:, 0].astype(int)
    data = raw_data[:, 1:]
    # print(label[:5], data[:5, :10])
    return label, data


def loadd(p: path.Path):
    data = []
    label = []
    with open(str(p.absolute()), 'r') as f:
        for line in f.readlines():
            a = line.split(',')
            label.append(int(a[0]))
            data.append(np.array(a[1:]).astype(float))
    return label, data


def join_path(data_dir, name):
    p = os.path.join(data_dir, name)
    p_train = os.path.join(p, f"{name}_TRAIN")
    p_test = os.path.join(p, f"{name}_TEST")
    return path.Path(p_train), path.Path(p_test)


def load_ucr(data_dir, name):
    train_path, test_path = join_path(data_dir, name)
    train_label, train_data = load(train_path)
    test_label, test_data = load(test_path)
    return train_data, train_label, test_data, test_label


def load_ucrd(data_dir, name):
    train_path, test_path = join_path(data_dir, name)
    train_label, train_data = loadd(train_path)
    test_label, test_data = loadd(test_path)
    return train_data, train_label, test_data, test_label

if __name__ == "__main__":
    p = path.Path('../data/ucr2015_d/50words/50words_TEST')
    loadd(p)

