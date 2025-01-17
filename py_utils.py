import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
"""
将数据
"""

class random_mini_batches_standardtwoModality(Dataset):
    def __init__(self, x1_data, x2_data, y_data):
        self.x1_data = x1_data
        self.x2_data = x2_data
        self.y_data = y_data

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        x1 = self.x1_data[index]
        x2 = self.x2_data[index]
        y = self.y_data[index]
        y = y.squeeze().tolist()
        return x1, x2, y


class  random_mini_batches(Dataset):
    def __init__(self, x1_data, x2_data,x1_full_data,x2_full_data, y_data):
        self.x1_data = x1_data
        self.x2_data = x2_data
        self.x1_full_data = x1_full_data
        self.x2_full_data = x2_full_data
        self.y_data = y_data


    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        x1 = self.x1_data[index]
        x2 = self.x2_data[index]
        x1_full = self.x1_full_data[index]
        x2_full = self.x2_full_data[index]
        y = self.y_data[index]
        return x1, x2,x1_full,x2_full,y

def mirror_concatenate(x, max_hw=3):
    # 给下面和右侧填充像素
    #x_extension = cv2.copyMakeBorder(x, 0, max_hw, 0, max_hw, cv2.BORDER_REFLECT)
    x_extension = cv2.copyMakeBorder(x, max_hw, max_hw, max_hw, max_hw, cv2.BORDER_REFLECT)
    return x_extension

def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

def convert_to_one_hot(Y, C):
    # 都热编码
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def generate_batch(idx, X_PCAMirrow, Y, batch_size, ws,row,col, shuffle=False):
    num = len(idx)
    if shuffle:
        np.random.shuffle(idx)

    for i in range(0, num, batch_size):
        bi = np.array(idx)[np.arange(i, min(num, i + batch_size))]# bi相当于批次的索引
        index_row = np.ceil((bi + 1) * 1.0 / col).astype(np.int32)# 找到每个批次对应的行号
        index_col = (bi + 1) - (index_row - 1) * col # 找到每个批次对应的列号（已知二维矩阵的行号，求二维矩阵的列号）
        # index_row += hw - 1
        # index_col += hw - 1
        patches = np.zeros([bi.size, ws*ws*X_PCAMirrow.shape[-1]])
        # .shape[-1]代表最后一个维度的大小也就是144
        # 每个批次对应一个patch,总批次个的7*7*144=7056，一行相当于一个patch的像素特征

        for j in range(bi.size):
            a = index_row[j] - 1  # hw  找到当前批次对应的行号
            b = index_col[j] - 1  # hw  找到当前批次对应的列号
            # 访问数组元素，索引从0开始
            patch = X_PCAMirrow[a:a + ws, b:b + ws, :]
            # 从X_PCAMirrow中提取一个大小为ws*ws的patch，起始点为(a,b)。
            # 这行代码的含义是提取X_PCAMirrow中以(a,b)为左上角，wsws个像素组成的patch，
            # 其中包含了X_PCAMirrow.shape[-1]个通道（即颜色通道数）
            patches[j, :] = patch.reshape(ws*ws*X_PCAMirrow.shape[-1])
            #取出patches的第j行的所有列，将patch放到patches第j行里

        labels = Y[bi]
        # Y[bi]返回的是Y中索引为bi的那些元素构成的数组，即当前batch中样本对应的标签数组
        labels[labels==0]=1
        labels = convert_to_one_hot(labels-1, 15)
        labels = labels.T

        yield patches,labels

def sampling(Y_train,Y_test):
    n_class = Y_test.max() # Y_test是一个一维数组，所以可以索引出最大值。
    train_idx = list()
    test_idx = list()

    for i in range(1, n_class + 1):
        train_i = np.where(Y_train == i)[0] # np.where()获得的是一个元组，但是我们需要获得一个数组，所以用[0]
        test_i = np.where(Y_test == i)[0]

        train_idx.extend(train_i) # 将数组中的元素逐一添加到列表里
        test_idx.extend(test_i)

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    return train_idx, test_idx

def generate_cube(idx, X, Y, ws,row,col, shuffle=False):
    num = len(idx)   #2832
    if shuffle:
        np.random.shuffle(idx)

    bi = np.array(idx) # 重新创建一个bi，这样操作bi不会影响到idx
    index_row = np.ceil((bi + 1)/ col).astype(np.int32) # bi+1相当于对bi里的每一个数+1（np是从0开始索引的）
    index_col = (bi + 1) - (index_row - 1) * col # 数组是从0开始的，0号所以实际是第一个位置。所以bi+1
    patches = np.zeros([bi.size, ws*ws*X.shape[-1]]) # bi.size是行数，ws*ws*X.shape[-1]=7*7*144
    # 每个批次对应一个patch,总批次个的7*7*144=7056，一行相当于一个patch的像素特征
    for j in range(bi.size):
        a = index_row[j] - 1  # 数组索引从0开始
        b = index_col[j] - 1
        patch = X[a:a + ws, b:b+ws, :]
        patches[j, :] = patch.reshape(ws*ws*X.shape[-1])
    labels = Y[bi]-1
    # 转为one-hot之前需要将label都-1才能使用
    labels = convert_to_one_hot(labels, 15)
    labels = labels.T

    return patches,labels