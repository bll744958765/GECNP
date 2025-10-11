# -*- coding : utf-8 -*-

"""
数据生成
"""

from torch.utils.data import Dataset

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import torch
import os

print(torch.version.cuda)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def normal(x, min_val=0):
    x_min = np.min(x)
    x_max = np.max(x)
    x_mean = np.mean(x)
    x_std = np.std(x) + 1e-8
    if x_min == 0 and x_max == 0:
        return x
    if min_val == -1:
        x_norm = 2 * ((x - x_min) / (x_max - x_min)) - 1
    if min_val == 0:
        # scaler=StandardScaler()
        # x_norm = scaler.fit_transform(x)
        # x_norm = ((x - x_min) / (x_max - x_min))
        x_norm = (x - x_mean) / x_std
    return x_norm


def split(xuhao, seed, dest):
    global test_size
    set_seed(seed=seed)
    if dest == 'generation':
        data_frame = pd.read_csv(r'./data/generated_data_with_target_T=2.csv')
        test_size = 0.9
    if dest == 'generation1':
        data_frame = pd.read_csv(r'./data/generated_data_with_target_T=1.csv')
        test_size = 0.9
    if dest == 'generation3':
        data_frame = pd.read_csv(r'./data/generated_data_with_target_T=3.csv')
        test_size = 0.9
    if dest == 'cali':
        data_frame = pd.read_csv(r'./data/fetch_california_housing.csv')
        test_size = 0.9
    if dest == 'temperature':
        data_frame = pd.read_csv(r'./data/1-4_mean.csv')
        test_size = 0.9
    if dest == 'Chengdu_housing':
        data_frame = pd.read_csv(r'./data/Chengdu_housing.csv')
        test_size = 0.9
    all_data = np.array(data_frame)

    for i in range(xuhao):
        # dest="{}".format(dest)
        save_path = "./trained/data/{}".format(dest)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if dest == 'cali':
            train_data_orgin, valid_data_orgin = train_test_split(all_data, test_size=test_size, random_state=seed)
            test_data_orgin, valid_data_orgin = train_test_split(valid_data_orgin, test_size=0.78, random_state=seed)
            train_data_orgin = pd.DataFrame(train_data_orgin)
            valid_data_orgin = pd.DataFrame(valid_data_orgin)
            test_data_orgin = pd.DataFrame(test_data_orgin)
            train_data_orgin.to_csv(save_path + '/train{}_orgin.csv'.format(i), index=False)
            valid_data_orgin.to_csv(save_path + '/valid{}_orgin.csv'.format(i), index=False)
            test_data_orgin.to_csv(save_path + '/test{}_orgin.csv'.format(i), index=False)
        else:
            train_data_orgin, valid_data_orgin = train_test_split(all_data, test_size=test_size, random_state=seed)
            train_data_orgin = pd.DataFrame(train_data_orgin)
            valid_data_orgin = pd.DataFrame(valid_data_orgin)
            train_data_orgin.to_csv(save_path + '/train{}_orgin.csv'.format(i), index=False)
            valid_data_orgin.to_csv(save_path + '/valid{}_orgin.csv'.format(i), index=False)

    y = all_data[:, -1]
    x = all_data[:, 1:-1]
    id = all_data[:, 0]
    if dest == 'Chengdu_housing':
        for i in range(3):
            x[:, i] = normal(x[:, i], min_val=0)
    else:
        for i in range(x.shape[1]):
            x[:, i] = normal(x[:, i], min_val=0)

    all_data = np.concatenate((np.concatenate((id.reshape(-1, 1), x), axis=-1), y.reshape(-1, 1)), axis=-1)
    for i in range(xuhao):
        save_path = "./trained/data/{}".format(dest)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if dest == 'cali':
            train_data, valid_data = train_test_split(all_data, test_size=test_size, random_state=seed)
            test_data, valid_data = train_test_split(valid_data, test_size=0.78, random_state=seed)

            train_data = pd.DataFrame(train_data)
            valid_data = pd.DataFrame(valid_data)
            test_data = pd.DataFrame(test_data)

            train_data.to_csv(save_path + '/train{}.csv'.format(i), index=False)
            valid_data.to_csv(save_path + '/valid{}.csv'.format(i), index=False)
            test_data.to_csv(save_path + '/test{}.csv'.format(i), index=False)
        else:
            train_data, valid_data = train_test_split(all_data, test_size=test_size, random_state=seed)

            train_data = pd.DataFrame(train_data)
            valid_data = pd.DataFrame(valid_data)

            train_data.to_csv(save_path + '/train{}.csv'.format(i), index=False)
            valid_data.to_csv(save_path + '/valid{}.csv'.format(i), index=False)
    return print('train_data:', len(train_data), 'valid_data：', len(valid_data))


def data_load(tt, dest):
    save_path = "./trained/data/{}".format(dest)
    train_data = pd.read_csv(save_path + '/train{}.csv'.format(tt))
    valid_data = pd.read_csv(save_path + '/valid{}.csv'.format(tt))

    train_data = np.array(train_data)
    valid_data = np.array(valid_data)
    if dest == 'generation':
        train_x = train_data[:, 1:3]
        train_y = train_data[:, (0,7)]
        valid1_x = valid_data[:, 1:3]
        valid_y = valid_data[:, (0, 7)]
    if dest == 'generation1':
        train_x = train_data[:, 1:3]
        train_y = train_data[:, (0,7)]
        valid1_x = valid_data[:, 1:3]
        valid_y = valid_data[:, (0, 7)]

    if dest == 'generation3':
        train_x = train_data[:, 1:3]
        train_y = train_data[:, (0,7)]
        valid1_x = valid_data[:, 1:3]
        valid_y = valid_data[:, (0, 7)]


    if dest == 'cali':
        train_x = train_data[:, 1:9]
        train_y = train_data[:, (0, 9)]
        valid1_x = valid_data[:, 1:9]
        valid_y = valid_data[:, (0, 9)]

    if dest == 'temperature':
        train_x = train_data[:, 1:13]
        train_y = train_data[:, (0, 13)]
        valid1_x = valid_data[:, 1:13]
        valid_y = valid_data[:, (0, 13)]

    if dest == "Chengdu_housing":
        train_x = train_data[:, 1:14]
        train_y = train_data[:, (0, 14)]
        valid1_x = valid_data[:, 1:14]
        valid_y = valid_data[:, (0, 14)]

    y = torch.tensor(train_y).to(device).float()
    valid_y = torch.tensor(valid_y).to(device).float().squeeze(-1)
    x = torch.FloatTensor(train_x).to(device).float()
    valid_x = torch.FloatTensor(valid1_x).to(device).float()

    return x, y, valid_x, valid_y


def get_tensor_from_pd(dataframe_series):
    return torch.tensor(data=dataframe_series.values)


class DatasetGP(Dataset):
    def __init__(self, n_tasks, xuhao, dest,
                 batch_size,
                 n_context_min,
                 n_context_max,n_target_max ):  #chengduhousing: n_context_max=1100, n_target_max=1480,
                                        #generation: n_context_max=200, n_target_max=250,

        super().__init__()
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.xuhao = xuhao
        self.n_context_min = n_context_min
        self.n_context_max = n_context_max
        self.n_target_max = n_target_max
        self.dest = dest

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, index):
        n_context = np.random.randint(self.n_context_min, self.n_context_max + 1)
        n_target = n_context + np.random.randint(3, self.n_target_max - n_context + 1)

        batch_context_x = []
        batch_context_y = []
        batch_target_x = []
        batch_target_y = []

        for _ in range(self.batch_size):
            x, y, _, _, = data_load(self.xuhao, self.dest)
            context_x = x[0: n_context, :]
            context_y = y[0: n_context, :]

            target_x = x[0:n_target, :]
            target_y = y[0:n_target, :]
            # target_x = x
            # target_y = y

            batch_context_x.append(context_x)
            batch_context_y.append(context_y)

            batch_target_x.append(target_x)
            batch_target_y.append(target_y)

        batch_context_x = torch.stack(batch_context_x, dim=0)
        batch_context_y = torch.stack(batch_context_y, dim=0)
        batch_target_x = torch.stack(batch_target_x, dim=0)
        batch_target_y = torch.stack(batch_target_y, dim=0)

        return batch_context_x, batch_context_y, batch_target_x, batch_target_y


class DatasetGP_test(Dataset):
    def __init__(self, n_tasks, xuhao, dest, batch_size=1):
        super().__init__()
        self.n_tasks = n_tasks
        self.xuhao = xuhao
        self.batch_size = batch_size
        self.dest = dest

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, index):
        batch_context_x = []
        batch_context_y = []
        batch_target_x = []
        batch_target_y = []

        for _ in range(self.batch_size):
            x, y, valid_x, valid_y = data_load(self.xuhao, self.dest)
            context_x = x
            context_y = y

            target_x = valid_x
            target_y = valid_y

            batch_context_x.append(context_x)
            batch_context_y.append(context_y)

            batch_target_x.append(target_x)
            batch_target_y.append(target_y)

        batch_context_x = torch.stack(batch_context_x, dim=0)
        batch_context_y = torch.stack(batch_context_y, dim=0)
        batch_target_x = torch.stack(batch_target_x, dim=0)
        batch_target_y = torch.stack(batch_target_y, dim=0)

        return batch_context_x, batch_context_y, batch_target_x, batch_target_y
