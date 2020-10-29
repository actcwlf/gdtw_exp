from auxiliary.dataset import load_ucrd
import torch
from torch import nn
from torch.functional import F
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
from tqdm import tqdm
import fsdtw
from modules.data_gen import equal_interp
from modules.gsfdtw_module import GFSDTWLayer


class GDTWNet(nn.Module):
    def __init__(self, n_fsdtw, input_len, target_class, gamma, q, radius):
        super().__init__()
        self.n_fsdtw = n_fsdtw
        self.target_class = target_class

        self.fsdtwlayer = nn.ModuleList([
            GFSDTWLayer(input_len, k_len=100, gamma=gamma, q=q, radius=radius) for i in range(n_fsdtw)
        ])

        output_len = self.n_fsdtw * self.fsdtwlayer[0].output_len
        inner_width = max(output_len, target_class) * 2

        # self.filter = nn.Sequential(
        #     nn.BatchNorm1d(output_len),
        #     nn.ReLU()
        # )

        self.mlp = nn.Sequential(
            # nn.BatchNorm1d(output_len),
            nn.Linear(output_len, inner_width),
            nn.BatchNorm1d(inner_width),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(inner_width, inner_width),
            nn.BatchNorm1d(inner_width),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(inner_width, target_class)
        )

    def forward(self, x):
        xs = [layer(x).float() for layer in self.fsdtwlayer]
        # xs.append(self.extractor(x).squeeze())
        xs = torch.cat(xs, dim=1)
        # xs = self.bn1(xs)
        # xs = torch.relu(xs)
        # xs = self.filter(xs)
        tx = self.mlp(xs)
        tx = F.softmax(tx, dim=1)
        return tx


class RefNet(nn.Module):
    def __init__(self, input_len, target_class):
        super().__init__()
        self.target_class = target_class



        output_len = int(np.log2(input_len)) - 1
        inner_width = max(output_len, target_class) * 2

        self.extractor = nn.Sequential(
            nn.Linear(input_len, 2),
            # nn.BatchNorm1d(input_len),
            nn.ReLU(),
            nn.Linear(2, output_len),
            # nn.BatchNorm1d(input_len),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            # nn.BatchNorm1d(output_len),
            nn.Linear(output_len, inner_width),
            nn.BatchNorm1d(inner_width),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(inner_width, inner_width),
            nn.BatchNorm1d(inner_width),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(inner_width, target_class)
        )

    def forward(self, x):
        # t = self.linear(x)
        # tx = self.extractor(x.reshape(-1, 1, 270)).squeeze()
        tx = self.extractor(x).squeeze()
        tx = self.mlp(tx)
        tx = F.softmax(tx, dim=1)
        return tx

class UCRDDataset(Dataset):
    def __init__(self, x, y, interp=False, interp_len=None):
        self.x = None
        self.min_len = self.max_len = len(x[0])
        for xx in x:
            self.min_len = min(self.min_len, len(xx))
            self.max_len = max(self.max_len, len(xx))
        if interp_len is None:
            interp_len = int((self.min_len + self.max_len) / 2)
        self.interp_len = interp_len
        self.interp = interp
        if interp:
            data = [equal_interp(xx, self.interp_len) for xx in x]
            self.x = torch.tensor(data).float()
            self.x = (self.x - self.x.mean(dim=1, keepdim=True)) / self.x.std(dim=1, keepdim=True)
            # self.min_len = self.max_len = 270
        else:
            self.x = []
            self.min_len = self.max_len = len(x[0])
            for xx in x:
                xx = (xx - xx.mean()) / xx.std()
                if len(xx) < self.min_len:
                    self.min_len = len(xx)
                elif len(xx) > self.max_len:
                    self.max_len = len(xx)
                self.x.append(xx)
        self.y = torch.tensor(y).long() - 1 # ucr 类别是从1开始


    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def make_data_loader(dataset, batch_size, shuffle=False):
    if dataset.interp:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        return DDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class DDataLoader:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.perm = np.random.permutation(range(len(self.dataset)))
        else:
            self.perm = list(range(len(self.dataset)))
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.dataset):
            start = self.idx
            end = min([start + self.batch_size, len(self.dataset)])
            rg = self.perm[start:end]
            self.idx = end
            result = list(zip(*[self.dataset[i] for i in rg]))
            return list(result[0]), torch.tensor(result[1])
        else:
            raise StopIteration  # 抛出异常停止遍历


def exp(data_name, n_extractors, gamma, q, r, lr, epoch):
    X_tr, y_tr, X_te, y_te = load_ucrd("data/ucr2015_d", data_name)
    target_class = len(np.unique(y_tr))
    interp = False
    if SETTINGS['model'] == 'mlp':
        interp = True
    train_data = UCRDDataset(X_tr, y_tr, interp=interp)
    test_data = UCRDDataset(X_te, y_te, interp=interp, interp_len=train_data.interp_len)

    train_loader = make_data_loader(train_data, batch_size=16, shuffle=True)
    test_loader = make_data_loader(test_data, batch_size=32)

    min_len = min(train_data.min_len, test_data.min_len)
    if interp:
        model = RefNet(train_data.interp_len, target_class)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        model = GDTWNet(n_extractors, min_len, target_class, gamma, q, r)
        optim = torch.optim.Adam([
            {'params': model.fsdtwlayer.parameters(), 'lr': lr * 10},
            {'params': model.mlp.parameters(), 'lr': lr},
            # {'params': model.extractor.parameters(), 'lr': 5e-4}
        ])
    #
    print(model)
    loss_func = nn.CrossEntropyLoss()

    for i in range(epoch):
        total_loss = 0
        n_correct = 0
        model.train()
        for x, y in tqdm(train_loader, disable=True):
            # x = train_data.x[:]
            # y = train_data.y[:]
            model.zero_grad()
            output = model(x)
            loss = loss_func(output, y)
            pred = output.argmax(dim=1)
            n_correct += (pred == y).sum().item()
            loss.backward()
            grad_norm = optim.step()
            total_loss += loss.detach().numpy()
        print(f'epoch {i}, loss {total_loss}, train acc {n_correct, len(train_data)}')

        if (i+1) % 10 != 0:
            continue
        model.eval()
        total_loss = 0
        n_correct = 0
        for x, y in tqdm(test_loader, disable=True):
            # x = test_data.x
            # y = test_data.y
            output = model(x)
            loss = loss_func(output, y)
            pred = output.argmax(dim=1)
            n_correct += (pred == y).sum().item()
            total_loss += loss.detach().numpy()
        print(f'test, loss {total_loss}, test acc {n_correct, len(test_data ), n_correct / len(test_data )}')

SETTINGS = {
    "version": fsdtw.__version__,
    "data":'CBF',
    "model": "gdtw",
    "n_extractors": 1,
    "gamma": 0.2,
    "q": 20,
    "radius": 5,
    "lr": 0.01,
    "epoch": 500
}

if __name__ == "__main__":
    print(SETTINGS)
    exp(SETTINGS["data"],
        SETTINGS["n_extractors"],
        SETTINGS["gamma"],
        SETTINGS["q"],
        SETTINGS["radius"],
        SETTINGS["lr"],
        SETTINGS["epoch"])
