import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import random, os
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import pandas as pd
import copy

from fedlab.utils.dataset import FMNISTPartitioner, CIFAR10Partitioner
from fedlab.utils.functional import partition_report, save_dict
from fedlab.utils.dataset.rspartition import RSPartitioner

from args_cifar10_c2 import args_parser
import server_se1 as server
import model

from utils.global_test import test_on_globaldataset, globalmodel_test_on_localdataset, globalmodel_test_on_specifdataset
from utils.local_test import test_on_localdataset
from utils.training_loss import train_loss_show, train_localacc_show
from utils.sampling import testset_sampling, trainset_sampling, trainset_sampling_label
from utils.tSNE import FeatureVisualize
from py_utils import mirror_concatenate, same_seeds, generate_batch, sampling, generate_cube, random_mini_batches_standardtwoModality

args = args_parser()
def seed_torch(seed=args.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()
GLOBAL_SEED = 2#1
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    from torchaudio.utils.sox_utils import set_seed
    set_seed(GLOBAL_SEED + worker_id)

similarity = True
save_models = False
Train_model = True

# 数据加载和预处理
MODEL ='MML'
Train = True

HSI_TrSet = sio.loadmat('likyou_data/HSI_TrSet_reshaped.mat')['Data']# 加载数据集处理好的数据集
print(HSI_TrSet .shape)
HSI_TeSet = sio.loadmat('likyou_data/HSI_TeSet_reshaped.mat')['Data']
print(HSI_TeSet.shape)
LiDAR_TrSet = sio.loadmat('likyou_data/LiDAR_TrSet_reshaped.mat')['Data']
print(LiDAR_TrSet.shape)
LiDAR_TeSet = sio.loadmat('likyou_data/LiDAR_TeSet_reshaped.mat')['Data']
print(LiDAR_TeSet.shape)
Y_train = sio.loadmat('likyou_data/Y_train_reshaped.mat')['Data']
print(Y_train.shape)
Y_test = sio.loadmat('likyou_data/Y_test_reshaped.mat')['Data']
print(Y_test.shape)

TrLabel = sio.loadmat('HSI_LiDAR_CNN/TrLabel.mat')['TrLabel']
TeLabel = sio.loadmat('HSI_LiDAR_CNN/TeLabel.mat')['TeLabel']

# 将标签转换为 PyTorch 张量
Y_train1 = torch.tensor(TrLabel - 1, dtype=torch.long)
Y_test1 = torch.tensor(TeLabel - 1, dtype=torch.long)


HSI_MapSet = sio.loadmat('likyou_data/HSI.mat')['HSI']# 加载HSI整图
ground_truth=sio.loadmat('likyou_data/gt.mat')['gt']# 加载标签整图
LiDAR_MapSet = sio.loadmat('houston_full_new/Houstonlidar.mat')['LiDAR']
[row, col, n_feature] = HSI_MapSet.shape


#HSI整图数据标准化
HSI_MapSet = HSI_MapSet.reshape(row * col, n_feature)
HSI_MapSet = np.asarray(HSI_MapSet, dtype=np.float32)
HSI_MapSet=(HSI_MapSet-np.min(HSI_MapSet))/(np.max(HSI_MapSet)-np.min(HSI_MapSet))
HSI_MapSet = HSI_MapSet.reshape(row, col, n_feature)
HSI_MapSet = mirror_concatenate(HSI_MapSet)

#lidar整图数据标准化
[row, col, n_feature] = LiDAR_MapSet.shape
LiDAR_MapSet = LiDAR_MapSet.reshape(row * col, n_feature)
LiDAR_MapSet = np.asarray(LiDAR_MapSet, dtype=np.float32)
LiDAR_MapSet = (LiDAR_MapSet-np.min(LiDAR_MapSet))/(np.max(LiDAR_MapSet)-np.min(LiDAR_MapSet))
LiDAR_MapSet = LiDAR_MapSet.reshape(row, col, n_feature)
LiDAR_MapSet = mirror_concatenate(LiDAR_MapSet)


if MODEL.strip() == 'MML':
    trainset1=HSI_TrSet
    trainset2 = LiDAR_TrSet
    valset1=HSI_TeSet
    valset2 = LiDAR_TeSet
    Train = False


if MODEL.strip() ==  'CML-LiDAR':
    trainset1=HSI_TrSet
    trainset2 = LiDAR_TrSet
    valset1=np.zeros_like(HSI_TeSet)
    valset2 = LiDAR_TeSet
    Train = True

if MODEL.strip() == 'CML-HSI':
    trainset1=HSI_TrSet
    trainset2 = LiDAR_TrSet
    valset1=HSI_TeSet
    valset2 = np.zeros_like(LiDAR_TeSet)
    Train = True

#转为tensor才能计算
X1_train = torch.tensor(trainset1)
X2_train = torch.tensor(trainset2)
X1_test = torch.tensor(valset1)
X2_test = torch.tensor(valset2)

trainset = random_mini_batches_standardtwoModality(X1_train, X2_train, Y_train1)
testset = random_mini_batches_standardtwoModality(X1_test, X2_test, Y_test1)


specf_model = model.Cross_fusion_CNN_avg(144,21,15).to(args.device)
num_classes = args.num_classes
num_clients = args.K
number_perclass = args.num_perclass

col_names = [f"class{i}" for i in range(num_classes)]
print(col_names)
hist_color = '#4169E1'
plt.rcParams['figure.facecolor'] = 'white'
### Distribution-based (class)


Y_train_targets = Y_train1.squeeze().tolist()
Y_test_targets = Y_test1.squeeze().tolist()

# perform partition
noniid_labeldir_part = RSPartitioner(Y_train_targets,
                                num_clients = num_clients,
                                balance = None,
                                #partition="shards",
                                #num_shards=200,
                                partition = "dirichlet",
                                dir_alpha = 0.2,
                                seed = 1)
# generate partition report
csv_file = "./data/Houston2013/houston2013_noniid_labeldir_clients_10.csv"   ##读取CSV文件

noniid_labeldir_part_df = pd.read_csv(csv_file, header=1)
noniid_labeldir_part_df = noniid_labeldir_part_df.set_index('client')  ##将客户端ID设为索引
for col in col_names:
    noniid_labeldir_part_df[col] = (noniid_labeldir_part_df[col] * noniid_labeldir_part_df['Amount']).astype(int)
##最终得到的 noniid_labeldir_part_df 中的每个元素都代表了特定客户端中，属于特定类别的样本数量

# split dataset into training and testing

trainset_sample_rate = args.trainset_sample_rate
rare_class_nums = 0
trainset.targets = Y_train_targets
testset.targets = Y_test_targets
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
val_loader = DataLoader(testset, batch_size=len(testset), shuffle=True)

dict_users_train = trainset_sampling_label(args, trainset, trainset_sample_rate, rare_class_nums, noniid_labeldir_part)
dict_users_test = testset_sampling(args, testset, number_perclass, noniid_labeldir_part_df)


# perform partition
iid_part = RSPartitioner(Y_train[0],
                            num_clients=num_clients,
                            partition="iid",
                            seed=1)
# generate partition report
csv_file = "./data/Houston2013/houston2013_iid_labeldir_clients_10.csv"

iid_part_df = pd.read_csv(csv_file, header=1)
iid_part_df = iid_part_df.set_index('client')
for col in col_names:
    iid_part_df[col] = (iid_part_df[col] * iid_part_df['Amount']).astype(int)


dict_users_train_iid = trainset_sampling_label(args, trainset, trainset_sample_rate, rare_class_nums, iid_part)
dict_users_test_iid = testset_sampling(args, testset, number_perclass, iid_part_df)

training_number = {j: {} for j in range(args.K)}

for i in range(args.K):
    training_number[i] = {j: 0 for j in range(num_classes)}
    label_class = set(np.array(trainset.targets)[list(dict_users_train_iid[i])].tolist())
    # print(list(label_class))
    for k in label_class:
        training_number[i][k] = list(np.array(trainset.targets)[list(dict_users_train_iid[i])]).count(k)


# 示例数据，替换为你的实际训练集数据
# train_set = np.random.randint(0, 10, size=(2832, 15))
# 此处我们假设类别标签在最后一列

# # 读取实际数据（这里假设最后一列是类别标签）
# labels = testset.targets
# # 如果类别标签是浮点数，先转换为整数
#
# # 统计每个类别的样本数量
# unique, counts = np.unique(labels, return_counts=True)
# class_counts = dict(zip(unique, counts))
#
# # 打印每个类别的样本数量
# for class_label, count in class_counts.items():
#     print(f"类别 {class_label} 的样本数量: {count}")


# initiate the server with defined model and dataset
serverz = server.Server(args, specf_model, trainset, dict_users_train) #dict_users指的是user的local dataset索引

#  baseline----> n-iid setting with fedavg
server_fedavg = copy.deepcopy(serverz)#dict_users指的是user的local dataset索引

if Train_model:
    global_model1, similarity_dict1, client_models1, loss_dict1, clients_index1, acc_list1 = server_fedavg.fedavg_joint_update(testset, dict_users_test_iid[0],similarity = similarity,test_global_model_accuracy = True)

else:
    if similarity:
        similarity_dict1 = torch.load("results/Test/label skew/cifar10/fedavg/seed{}/similarity_dict1_{}E_{}class.pt".format(args.seed,args.E,C))
    acc_list1 = torch.load("results/Test/label skew/cifar10/fedavg/seed{}/acc_list1_{}E_{}class.pt".format(args.seed,args.E,C))
    global_model1 = server_fedavg.nn
    client_models1 = server_fedavg.nns
    path_fedavg = "results/Test/label skew/cifar10/fedavg/seed{}/global_model_fedavg_{}E_{}class.pt".format(args.seed,args.E,C)
    global_model1.load_state_dict(torch.load(path_fedavg))
    for i in range(args.K):
        path_fedavg = "results/Test/label skew/cifar10/fedavg/seed{}/client{}_model_fedavg_{}E_{}class".format(args.seed,i,args.E,C)
        client_models1[i]=copy.deepcopy(global_model1)
        client_models1[i].load_state_dict(torch.load(path_fedavg))


g1,_ = test_on_globaldataset(args, global_model1, testset)
a1,_ =globalmodel_test_on_localdataset(args,global_model1, testset,dict_users_test)


#fedFA
# server_feature = copy.deepcopy(serverz)
#
# if Train_model:
#     (global_modelfa, similarity_dictfa, client_modelsfa,
#      loss_dictfa, clients_indexfa, acc_listfa) = server_feature.fedfa_anchorloss(testset,
#                                                                                  dict_users_test_iid[0],
#                                                                                  similarity = similarity,
#                                                                                  test_global_model_accuracy = True)
# else:
#     if similarity:
#         similarity_dictfa = torch.load("results/Test/label skew/cifar10/fedfa/seed{}/similarity_dictfa_{}E_{}class.pt".format(args.seed,args.E,C))
#     acc_listfa = torch.load("results/Test/label skew/cifar10/fedfa/seed{}/acc_listfa_{}E_{}class.pt".format(args.seed,args.E,C))
#     global_modelfa = server_feature.nn
#     client_modelsfa = server_feature.nns
#     path_fedfa = "results/Test/label skew/cifar10/fedfa/seed{}/global_model_fedfa_{}E_{}class".format(args.seed,args.E,C)
#     global_modelfa.load_state_dict(torch.load(path_fedfa))
#
#     for i in range(args.K):
#         path_fedfa = "results/Test/label skew/cifar10/fedfa/seed{}/client{}_model_fedfa_{}E_{}class".format(args.seed,i,args.E,C)
#         client_modelsfa[i] = copy.deepcopy(global_modelfa)
#         client_modelsfa[i].load_state_dict(torch.load(path_fedfa))
#
# save_models = False
# if save_models:
#     if similarity:
#         torch.save(similarity_dictfa,"results/Test/label skew/cifar10/iid-fedavg/seed{}/similarity_dictfa_{}E_{}class.pt".format(args.seed,args.E,C))
#     torch.save(acc_listfa,"results/Test/label skew/cifar10/fedfa/seed{}/acc_listfa_{}E_{}class.pt".format(args.seed,args.E,C))
#     df = pd.DataFrame(acc_listfa)
#     df.to_excel("results/Test/label skew/cifar10/fedfa/seed{}/acc_listfa_{}E_{}class.xlsx".format(args.seed,args.E,C))
#     path_fedfa = "results/Test/label skew/cifar10/fedfa/seed{}/global_model_fedfa_{}E_{}class".format(args.seed,args.E,C)
#     torch.save(global_modelfa.state_dict(), path_fedfa)
#
# gfa, _ = test_on_globaldataset(args, global_modelfa, testset)
#
# afa, _ =globalmodel_test_on_localdataset(args,global_modelfa, testset, dict_users_test)
# np.mean(list(afa.values()))
#
# if Train_model:
#     train_loss_show(args, loss_dictfa, clients_indexfa)