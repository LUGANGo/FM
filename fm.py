# -*- coding: utf-8 -*-
# 作者 ：LuGang
# 开发时间 ：2021/8/8 20:02
# 文件名称 ：fm.py
# 开发工具 ：PyCharm


import os
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
import matplotlib.pyplot as plt
from data_process import GetOrigData

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class FM(nn.Module):
    def __init__(self, n, k):
        super(FM, self).__init__()
        self.n = n  # 特征数
        self.k = k  # 因子数
        self.linear_part = nn.Linear(self.n, 1, bias=True)
        self.v = nn.Parameter(torch.rand(self.k, self.n))

    def fm(self, x):
        linear_part = self.linear_part(x)
        cross_part1 = torch.mm(x, self.v.t())
        cross_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t())
        cross_part = torch.sum(torch.sub(torch.pow(cross_part1, 2), cross_part2), dim=1)
        output = linear_part.transpose(1, 0) + 0.5 * cross_part

        return output

    def forward(self, x):
        output = self.fm(x)
        return output


def LoadData():
    datapath = os.path.join("orig_data.csv")
    if os.path.exists(datapath):
        return GetTrainTestData()
    else:
        # 先创建数据集
        data = GetOrigData()
        data.GetData()
        # 划分数据集
        data.split_train_test_data()
        return GetTrainTestData()


def GetTrainTestData():
    train_data_path = "train_data.csv"
    test_data_path = "test_data.csv"
    train_data = pd.read_csv(train_data_path, sep=",", header=0)
    # 将性别转化为one-hot编码
    train_data = pd.get_dummies(train_data.gender, drop_first=True).join(train_data)
    del train_data["gender"]

    test_data = pd.read_csv(test_data_path, sep=",", header=0)
    test_data = pd.get_dummies(test_data.gender, drop_first=True).join(test_data)
    del test_data["gender"]

    x_train = train_data.iloc[:, 0:len(train_data.columns) - 1].values
    y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
    x_test = test_data.iloc[:, 0:len(test_data.columns) - 1].values
    y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

    train_dataset = Data.TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                       torch.tensor(y_train, dtype=torch.float32))
    test_dataset = Data.TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                      torch.tensor(y_test, dtype=torch.float32))

    loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=False
    )

    # 测试
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        shuffle=False
    )

    return loader, test_loader, x_train.shape[1]


loader, test_loader, n = LoadData()
k = 5
epochs = 20
fm = FM(n, k)
learning_rate = 0.001
optimizer = torch.optim.SGD(fm.parameters(), lr=learning_rate)
loss_func = torch.nn.MSELoss()
loss_train_set = []
loss_test_set = []


def train():
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            output = fm(batch_x)
            output = output.transpose(1, 0)
            rmse_loss = torch.sqrt(loss_func(output, batch_y))

            l2_regularization = torch.tensor(0).float()
            for param in fm.parameters():
                l2_regularization += torch.norm(param, 2)

            loss = rmse_loss + l2_regularization
            loss.backward()
            optimizer.step()
        loss_train_set.append(loss)
        print("epoch(s): %d, train_loss: %f" % (epoch + 1, loss.item()))
        test_loss()

    draw(loss_train_set, loss_test_set)


def test_loss():
    with torch.no_grad():
        for (x, y) in test_loader:
            output = fm(x)
            output = output.transpose(1, 0)
            rmse_loss = torch.sqrt(loss_func(output, y))
        loss_test_set.append(rmse_loss)
        print("test_loss: %f" % rmse_loss)


def draw(loss_train_set, loss_test_set):
    x = [i for i in range(len(loss_train_set))]
    plt.plot(x, loss_train_set, label="Training loss")
    plt.plot(x, loss_test_set, label="Validation loss")
    plt.xlabel("epochs")
    plt.ylabel("rmse")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()
