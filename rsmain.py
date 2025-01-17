from __future__ import print_function
from __future__ import division

import os
from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    display_dataset,
    display_lidar_data,
    display_predictions,
    explore_spectrums,
    plot_spectrums,
    sample_gt,
    build_dataset,
    show_results,
    compute_imf_weights,
    get_device,
    padding_image,
    restore_from_padding,
    seed_torch,
)
from datasets import get_dataset, MultiModalX, open_file, DATASETS_CONFIG
import argparse

dataset_names = [
    v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--dataset", type=str, default=None, choices=dataset_names, help="Dataset to use."
)

parser.add_argument(
    "--folder",
    type=str,
    help="Folder where to store the "
    "datasets (defaults to the current working directory).",
    default="./Datasets/",)


# Dataset options
group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument(
    "--train_val_split",
    type=float,
    default=1,
    help="Percentage of samples to use for training and validation, "
         "'1' means all training data are used to train",
)
group_dataset.add_argument(
    "--training_sample",
    type=float,
    default=10,
    help="Percentage of samples to use for training (default: 10%%) and testing",
)
group_dataset.add_argument(
    "--sampling_mode",
    type=str,
    help="Sampling mode" " (random sampling or disjoint, default: random)",
    default="random",
)
group_dataset.add_argument(
    "--train_set",
    type=str,
    default=None,
    help="Path to the train ground truth (optional, this "
    "supersedes the --sampling_mode option)",
)
group_dataset.add_argument(
    "--test_set",
    type=str,
    default=None,
    help="Path to the test set (optional, by default "
    "the test_set is the entire ground truth minus the training)",
)
# Training options
group_train = parser.add_argument_group("Training")

group_train.add_argument(
    "--patch_size",
    type=int,
    help="Size of the spatial neighbourhood (optional, if "
    "absent will be set by the model)",
)
group_train.add_argument(
    "--lr", type=float, help="Learning rate, set by the model if not specified."
)
group_train.add_argument(
    "--class_balancing",
    action="store_true",
    help="Inverse median frequency class balancing (default = False)",
)
group_train.add_argument(
    "--batch_size",
    type=int,
    help="Batch size (optional, if absent will be set by the model",
)
group_train.add_argument(
    "--test_stride",
    type=int,
    default=1,
    help="Sliding window step stride during inference (default = 1)",
)
# Data augmentation parameters
group_da = parser.add_argument_group("Data augmentation")
group_da.add_argument(
    "--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)"
)
group_da.add_argument(
    "--radiation_augmentation",
    action="store_true",
    help="Random radiation noise (illumination)",
)
group_da.add_argument(
    "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
)

parser.add_argument(
    "--with_exploration", action="store_true", help="See data exploration visualization"
)
parser.add_argument(
    "--download",
    type=str,
    default=None,
    nargs="+",
    choices=dataset_names,
    help="Download the specified datasets and quits.",
)


args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
SAMPLE_TRAIN_VALID = args.train_val_split
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import random, os
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import pandas as pd
import copy

from fedlab.utils.dataset import FMNISTPartitioner, CIFAR10Partitioner
from fedlab.utils.functional import partition_report, save_dict

from args_cifar10_c2 import args_parser
import server_se1 as server
import model

from utils.global_test import test_on_globaldataset, globalmodel_test_on_localdataset, globalmodel_test_on_specifdataset
from utils.local_test import test_on_localdataset
from utils.training_loss import train_loss_show, train_localacc_show
from utils.sampling import testset_sampling, trainset_sampling, trainset_sampling_label
from utils.tSNE import FeatureVisualize

args = args_parser()


def seed_torch(seed=args.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()
GLOBAL_SEED = 2  # 1


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    from torchaudio.utils.sox_utils import set_seed
    set_seed(GLOBAL_SEED + worker_id)


similarity = True
save_models = False
Train_model = True

# C = "2CNN_2"
C = "resnet18"
# specf_model = model.Client_Model(args, name='cifar10').to(args.device)
specf_model = model.Cross_fusion_CNN(144,21,15).to(args.device)

root = "data/CIFAR10/"
trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=trans_cifar10)
testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=trans_cifar10)
num_classes = args.num_classes
num_clients = args.K
number_perclass = args.num_perclass

col_names = [f"class{i}" for i in range(num_classes)]
print(col_names)
hist_color = '#4169E1'
plt.rcParams['figure.facecolor'] = 'white'
### Distribution-based (class)

# perform partition
noniid_labeldir_part = CIFAR10Partitioner(trainset.targets,
                                          num_clients=num_clients,
                                          balance=None,
                                          # artition="shards",
                                          # num_shards=200,
                                          partition="dirichlet",
                                          dir_alpha=0.1,
                                          seed=1)
# generate partition report
csv_file = "data/CIFAR10/cifar10_noniid_labeldir_clients_10.csv"
partition_report(trainset.targets, noniid_labeldir_part.client_dict,
                 class_num=num_classes,
                 verbose=False, file=csv_file)

noniid_labeldir_part_df = pd.read_csv(csv_file, header=1)
noniid_labeldir_part_df = noniid_labeldir_part_df.set_index('client')
for col in col_names:
    noniid_labeldir_part_df[col] = (noniid_labeldir_part_df[col] * noniid_labeldir_part_df['Amount']).astype(int)

# select first 10 clients for bar plot
noniid_labeldir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
# plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('sample num')
plt.savefig(f"data/CIFAR10//cifar10_noniid_labeldir_clients_10.png",
            dpi=400, bbox_inches='tight')
# split dataset into training and testing

trainset_sample_rate = args.trainset_sample_rate
rare_class_nums = 0
dict_users_train = trainset_sampling_label(args, trainset, trainset_sample_rate, rare_class_nums, noniid_labeldir_part)
dict_users_test = testset_sampling(args, testset, number_perclass, noniid_labeldir_part_df)
# show the dataset split

training_number = {j: {} for j in range(args.K)}

for i in range(args.K):
    training_number[i] = {j: 0 for j in range(num_classes)}
    label_class = set(np.array(trainset.targets)[list(dict_users_train[i])].tolist())
    # print(list(label_class))
    for k in label_class:
        training_number[i][k] = list(np.array(trainset.targets)[list(dict_users_train[i])]).count(k)

df_training_number = []
df_training_number = pd.DataFrame(df_training_number)
for i in range(args.K):
    temp = pd.Series(training_number[i])
    df_training_number[i] = temp

df_training_number['Col_sum'] = df_training_number.apply(lambda x: x.sum(), axis=1)
df_training_number.loc['Row_sum'] = df_training_number.apply(lambda x: x.sum())

test_number = {j: {} for j in range(args.K)}

for i in range(args.K):
    test_number[i] = {j: 0 for j in range(num_classes)}
    label_class = set(np.array(testset.targets)[list(dict_users_test[i])].tolist())
    # print(list(label_class))
    for k in label_class:
        test_number[i][k] = list(np.array(testset.targets)[list(dict_users_test[i])]).count(k)

df_test_number = []
df_test_number = pd.DataFrame(df_test_number)
for i in range(args.K):
    temp = pd.Series(test_number[i])
    df_test_number[i] = temp

df_test_number['Col_sum'] = df_test_number.apply(lambda x: x.sum(), axis=1)
df_test_number.loc['Row_sum'] = df_test_number.apply(lambda x: x.sum())

# perform partition
iid_part = FMNISTPartitioner(trainset.targets,
                             num_clients=num_clients,
                             partition="iid",
                             seed=1)
# generate partition report
csv_file = "data/CIFAR10/cifar10_iid_clients_10.csv"
partition_report(trainset.targets, iid_part.client_dict,
                 class_num=num_classes,
                 verbose=False, file=csv_file)

iid_part_df = pd.read_csv(csv_file, header=1)
iid_part_df = iid_part_df.set_index('client')
for col in col_names:
    iid_part_df[col] = (iid_part_df[col] * iid_part_df['Amount']).astype(int)

# select first 10 clients for bar plot
iid_part_df[col_names].iloc[:10].plot.barh(stacked=True)
# plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('sample num')
plt.savefig(f"data/CIFAR10/cifar10_iid_clients_10.png",
            dpi=400, bbox_inches='tight')

dict_users_train_iid = trainset_sampling_label(args, trainset, trainset_sample_rate, rare_class_nums, iid_part)
dict_users_test_iid = testset_sampling(args, testset, number_perclass, iid_part_df)

training_number = {j: {} for j in range(args.K)}

for i in range(args.K):
    training_number[i] = {j: 0 for j in range(num_classes)}
    label_class = set(np.array(trainset.targets)[list(dict_users_train_iid[i])].tolist())
    # print(list(label_class))
    for k in label_class:
        training_number[i][k] = list(np.array(trainset.targets)[list(dict_users_train_iid[i])]).count(k)

df_training_number = []
df_training_number = pd.DataFrame(df_training_number)
for i in range(args.K):
    temp = pd.Series(training_number[i])
    df_training_number[i] = temp

df_training_number['Col_sum'] = df_training_number.apply(lambda x: x.sum(), axis=1)
df_training_number.loc['Row_sum'] = df_training_number.apply(lambda x: x.sum())

# initiate the server with defined model and dataset

serverz = server.Server(args, specf_model, trainset, dict_users_train)  # dict_users指的是user的local dataset索引

# #  baseline----> iid setting with fedavg
#
# server_iid = server.Server(args, specf_model, trainset, dict_users_train_iid)
# if Train_model:
#     global_model_iid, similarity_dict_iid, client_models_iid, loss_dict_iid, clients_index_iid, acc_list_iid = server_iid.fedavg_joint_update(testset, dict_users_test_iid[0], iid= True, similarity = similarity, test_global_model_accuracy = True)
# else:
#     if similarity:
#         similarity_dict_iid = torch.load("results/Test/label skew/cifar10/iid-fedavg/seed{}/similarity_dict_iid_{}E_{}class.pt".format(args.seed,args.E,C))
#     acc_list_iid = torch.load("results/Test/label skew/cifar10/iid-fedavg/seed{}/acc_list_iid_{}E_{}class.pt".format(args.seed,args.E,C))
#     global_model_iid = server_iid.nn
#     client_models_iid = server_iid.nns
#     path_iid_fedavg = "results/Test/label skew/cifar10/iid-fedavg/seed{}/global_model_iid-fedavg_{}E_{}class.pt".format(args.seed,args.E,C)
#     global_model_iid.load_state_dict(torch.load(path_iid_fedavg))
#     for i in range(args.K):
#         path_iid_fedavg = "results/Test/label skew/cifar10/iid-fedavg/seed{}/client{}_model_fedavg_{}E_{}class".format(args.seed,i,args.E,C)
#         client_models_iid[i]=copy.deepcopy(global_model_iid)
#
# if save_models:
#     if similarity:
#         torch.save(similarity_dict_iid,"results/Test/label skew/cifar10/iid-fedavg/seed{}/similarity_dict_iid_{}E_{}class.pt".format(args.seed,args.E,C))
#     torch.save(acc_list_iid,"results/Test/label skew/cifar10/iid-fedavg/seed{}/acc_list_iid_{}E_{}class.pt".format(args.seed,args.E,C))
#     path_iid_fedavg = "results/Test/label skew/cifar10/iid-fedavg/seed{}/global_model_iid-fedavg_{}E_{}class.pt".format(args.seed,args.E,C)
#     torch.save(global_model_iid.state_dict(), path_iid_fedavg)
#
# g_iid,_ = test_on_globaldataset(args, global_model_iid, testset)
#
#
# a_iid,_ =globalmodel_test_on_localdataset(args,global_model_iid, testset,dict_users_test_iid)
# np.mean(list(a_iid.values()))
#
# if Train_model:
#     train_loss_show(args, loss_dict_iid,clients_index_iid)
#
# server_fedavg =  copy.deepcopy(serverz)#dict_users指的是user的local dataset索引
#
# if Train_model:
#     global_model1, similarity_dict1, client_models1, loss_dict1, clients_index1, acc_list1 = server_fedavg.fedavg_joint_update(testset, dict_users_test_iid[0],similarity = similarity,test_global_model_accuracy = True)
# else:
#     if similarity:
#         similarity_dict1 = torch.load("results/Test/label skew/cifar10/fedavg/seed{}/similarity_dict1_{}E_{}class.pt".format(args.seed,args.E,C))
#     acc_list1 = torch.load("results/Test/label skew/cifar10/fedavg/seed{}/acc_list1_{}E_{}class.pt".format(args.seed,args.E,C))
#     global_model1 = server_fedavg.nn
#     client_models1 = server_fedavg.nns
#     path_fedavg = "results/Test/label skew/cifar10/fedavg/seed{}/global_model_fedavg_{}E_{}class.pt".format(args.seed,args.E,C)
#     global_model1.load_state_dict(torch.load(path_fedavg))
#     for i in range(args.K):
#         path_fedavg = "results/Test/label skew/cifar10/fedavg/seed{}/client{}_model_fedavg_{}E_{}class".format(args.seed,i,args.E,C)
#         client_models1[i]=copy.deepcopy(global_model1)
#         client_models1[i].load_state_dict(torch.load(path_fedavg))
#
# if save_models:
#     if similarity:
#         torch.save(similarity_dict1,"results/Test/label skew/cifar10/iid-fedavg/seed{}/similarity_dict1_{}E_{}class.pt".format(args.seed,args.E,C))
#     torch.save(acc_list1,"results/Test/label skew/cifar10/fedavg/seed{}/acc_list1_{}E_{}class.pt".format(args.seed,args.E,C))
#     path_fedavg = "results/Test/label skew/cifar10/fedavg/seed{}/global_model_fedavg_{}E_{}class.pt".format(args.seed,args.E,C)
#     torch.save(global_model1.state_dict(), path_fedavg)
#
#
# g1,_ = test_on_globaldataset(args, global_model1, testset)
# a1,_ =globalmodel_test_on_localdataset(args,global_model1, testset,dict_users_test)
# np.mean(list(a1.values()))
# if Train_model:
#     train_loss_show(args, loss_dict1,clients_index1)
# #train_localacc_show(args, mean_local_accuracy_list1)


# fedFA
server_feature = copy.deepcopy(serverz)

if Train_model:
    global_modelfa, similarity_dictfa, client_modelsfa, loss_dictfa, clients_indexfa, acc_listfa = server_feature.fedfa_anchorloss(
        testset, dict_users_test_iid[0],
        similarity=similarity,
        test_global_model_accuracy=True)
else:
    if similarity:
        similarity_dictfa = torch.load(
            "results/Test/label skew/cifar10/fedfa/seed{}/similarity_dictfa_{}E_{}class.pt".format(args.seed, args.E,
                                                                                                   C))
    acc_listfa = torch.load(
        "results/Test/label skew/cifar10/fedfa/seed{}/acc_listfa_{}E_{}class.pt".format(args.seed, args.E, C))
    global_modelfa = server_feature.nn
    client_modelsfa = server_feature.nns
    path_fedfa = "results/Test/label skew/cifar10/fedfa/seed{}/global_model_fedfa_{}E_{}class".format(args.seed, args.E,
                                                                                                      C)
    global_modelfa.load_state_dict(torch.load(path_fedfa))

    for i in range(args.K):
        path_fedfa = "results/Test/label skew/cifar10/fedfa/seed{}/client{}_model_fedfa_{}E_{}class".format(args.seed,
                                                                                                            i, args.E,
                                                                                                            C)
        client_modelsfa[i] = copy.deepcopy(global_modelfa)
        client_modelsfa[i].load_state_dict(torch.load(path_fedfa))

if save_models:
    if similarity:
        torch.save(similarity_dictfa,
                   "results/Test/label skew/cifar10/iid-fedavg/seed{}/similarity_dictfa_{}E_{}class.pt".format(
                       args.seed, args.E, C))
    torch.save(acc_listfa,
               "results/Test/label skew/cifar10/fedfa/seed{}/acc_listfa_{}E_{}class.pt".format(args.seed, args.E, C))
    df = pd.DataFrame(acc_listfa)
    df.to_excel("results/Test/label skew/cifar10/fedfa/seed{}/acc_listfa_{}E_{}class.xlsx".format(args.seed, args.E, C))
    path_fedfa = "results/Test/label skew/cifar10/fedfa/seed{}/global_model_fedfa_{}E_{}class".format(args.seed, args.E,
                                                                                                      C)
    torch.save(global_modelfa.state_dict(), path_fedfa)

gfa, _ = test_on_globaldataset(args, global_modelfa, testset)

afa, _ = globalmodel_test_on_localdataset(args, global_modelfa, testset, dict_users_test)
np.mean(list(afa.values()))

if Train_model:
    train_loss_show(args, loss_dictfa, clients_indexfa)