from auxiliary.dataset import load_ucr
from modules.gsfdtw_module import GFSDTWLayer
import torch
from torch import nn
from torch.functional import F
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
from tqdm import tqdm
import fsdtw
import time
# from torchstat import stat

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


class UCRDataset(Dataset):
    def __init__(self, x, y, min_id):
        self.x = torch.tensor(x).float()
        self.x = (self.x - self.x.mean(dim=1, keepdim=True)) / self.x.std(dim=1, keepdim=True)
        self.y = torch.tensor(y - min_id).long() # ucr 类别是从1开始

    def __getitem__(self, index):
        return self.x[index, :], self.y[index]

    def __len__(self):
        return self.x.shape[0]


def exp(data_name, n_extractors, gamma, q, r, lr, epoch):
    X_tr, y_tr, X_te, y_te = load_ucr("data/ucr2015", data_name)

    min_target_id = min(y_tr.min(), y_te.min())
    target_class = max(y_tr.max(), y_te.max()) - min_target_id + 1
    print('data len', X_tr.shape[0], 'class', target_class)
    train_data = UCRDataset(X_tr, y_tr, min_target_id)
    test_data = UCRDataset(X_te, y_te, min_target_id)

    batch_size = min([256, int(X_tr.shape[0] / 1.5)])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)

    model = GDTWNet(n_extractors, X_tr.shape[1], target_class, gamma, q, r)
    # model = FSDTWNetV2(n_extractors, X_tr.shape[1], target_class, gamma, r)
    # model = FSDTWNetV4(n_extractors, X_tr.shape[1], target_class, gamma, r)

    optim = torch.optim.Adam([
        {'params': model.fsdtwlayer.parameters(), 'lr': lr*10},
        {'params': model.mlp.parameters(), 'lr': lr},
        # {'params': model.extractor.parameters(), 'lr': 5e-4}
    ])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 200, 300, 400], gamma=0.1)
    loss_func = nn.CrossEntropyLoss()
    print(model)
    model.train()
    for i in range(epoch):
        total_loss = 0
        n_correct = 0
        ft = 0
        bt = 0

        for x, y in tqdm(train_loader, disable=True):
            # print(x.shape)
            model.zero_grad()

            ts = time.time()
            output = model(x)
            ft += time.time() - ts

            loss = loss_func(output, y)
            pred = output.argmax(dim=1)
            # t = pred == y
            # t = (pred == y).sum()
            n_correct += (pred == y).sum().item()

            ts = time.time()
            loss.backward()
            bt += time.time() - ts

            grad_norm = optim.step()
            # print(loss.detach().numpy())
            total_loss += loss.detach().numpy()/ x.shape[0]
        scheduler.step()
        print(f'epoch {i}\ntrain loss\t{total_loss:4f}\tacc\t{n_correct, len(train_loader.dataset)}\tforward {ft}\tbackward {bt}')
        total_loss = 0
        n_correct = 0
        if (i+1) % 10 != 0:
            continue
        model.eval()
        for x, y in tqdm(test_loader, disable=True):
            # print(x.shape)
            # model.zero_grad()
            output = model(x)
            loss = loss_func(output, y)
            pred = output.argmax(dim=1)
            # t = pred == y
            # t = (pred == y).sum()
            n_correct += (pred == y).sum().item()
            # loss.backward()
            # grad_norm = optim.step()
            # print(loss.detach().numpy())
            total_loss += loss.detach().numpy() / x.shape[0]
        model.train()
        print(f'test loss\t{total_loss:4f}\tacc\t{n_correct, len(test_loader.dataset), n_correct / len(test_loader.dataset)}')

"""
50words: 32, 1/0.1, 2, 0.001
Adiac: -
ArrowHead:
"""
SETTINGS = {
    "version": fsdtw.__version__,
    "data": 'Ham',
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
