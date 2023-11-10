import os
import pathlib
this_root_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(str(this_root_dir))

import os
import random
import shutil
import datetime
from copy import deepcopy

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

'''
############################################################################
                            hyperPamameters
############################################################################
'''
# class argsInput:
#     def __init__(self) -> None:
#         self.shuffle_times = 1
#         self.identity = 0.5
#         self.root_dir = "/tangzehua/tangzehua/RecCrossNet"
#         self.datasets_path = "{}/dataset/processed_data/pipeline_5/identity_{}_shuffle_{}".format(self.root_dir, self.identity, self.shuffle_times)
#         self.embedding_dir = "{}/dataset/esm_embedding_data/protein_700".format(self.root_dir)
#         self.out_dir = "/tangzehua/tangzehua/RecCrossNet/result/identity_{}_shuffle_{}".format(self.identity, self.shuffle_times)

class argsInput:
    def __init__(self) -> None:
        self.shuffle_times = 1
        self.identity = 0.8
        self.root_dir ='/lizutan/code/MCANet' #"/tangzehua/tangzehua/RecCrossNet"
        self.datasets_path = "{}/dataset/pipeline_5/identity_{}_shuffle_{}".format(self.root_dir, self.identity, self.shuffle_times)
        self.embedding_dir = "{}/preprocess/esm_embedding_data/protein_700".format(self.root_dir)
        self.out_dir = "/lizutan/code/MCANet/result/DrugBAN_model_identity_{}_shuffle_{}".format(self.identity, self.shuffle_times)

class hyperparameter:
    def __init__(self):
        self.Learning_rate = 1e-4
        self.Epoch = 100 #修改了epoch
        self.Batch_size = 256
        self.Patience = 50
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.protein_kernel = [4, 8, 12]
        self.drug_kernel = [4, 6, 8]
        self.conv = 128
        self.char_dim = 1280
        self.att_dim = 64
        self.loss_epsilon = 1
        self.num_workers = 16
        self.protein_max_len = 700
        self.attachment_site_max_len = 50
        self.mix_attention_head = 8

nucleotide_to_number = {
    "A": 1, "C": 2, "G": 3, "T": 4
}

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

'''
############################################################################
                            Helper Functions
############################################################################
'''

def get_datetime():
    '''output: month_day_hour_minute'''
    current_datetime = datetime.datetime.now()
    month = current_datetime.month
    day = current_datetime.day
    hour = current_datetime.hour
    minute = current_datetime.minute
    return "{}_{}_{}_{}".format(month, day, hour, minute)

def label_attachment_site(attB, attP, smi_ch_ind, MAX_HALF_SITE_LEN):
    # contact two att to 100 bp, and convert to index list(np)
    X = np.zeros(MAX_HALF_SITE_LEN * 2, np.int64())
    for i, ch in enumerate(attB[:MAX_HALF_SITE_LEN]):
        if ch in smi_ch_ind:
            X[i] = smi_ch_ind[ch]
    for i, ch in enumerate(attP[:MAX_HALF_SITE_LEN]):
        if ch in smi_ch_ind:
            X[i+MAX_HALF_SITE_LEN] = smi_ch_ind[ch]
    return X

def label_attachment_site_protein(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

class MakeLogClass:
    def __init__(self, log_file):
        self.log_file = log_file
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
    def make(self, *args):
        print(*args)
        # Write the message to the file
        with open(self.log_file, "a") as f:
            for arg in args:
                f.write("{}\r\n".format(arg))

'''
############################################################################
                        Dataset loadin functions
############################################################################
'''
def get_k_fold_datasets(args: argsInput, fold_index, k_fold=5):
    train_list = []
    for k_index in range(k_fold):
        k_table = pd.read_csv("{}/PASTE_{}_train_{}.txt".format(args.datasets_path, args.shuffle_times, k_index), sep="\t", header=None)
        if k_index == fold_index:
            validData = deepcopy(k_table)
        else:
            train_list.append(k_table)
    trainData = pd.concat(train_list, axis=0)
    return trainData, validData

class CustomDataset(Dataset):
    # 获取DNA的数字表示100，蛋白质的表征和标签x
    def __init__(self, data, embedding_dir, attachment_site_max_len=50):
        self.data = data
        self.attachment_site_max_len = attachment_site_max_len
        self.embedding_dir = embedding_dir
    
    # def rec_str_to_int(self, recombinases_id):
    #     rec_embedding = torch.load("{}/{}.pt".format(self.embedding_dir, recombinases_id))
    #     recombine_embedding = rec_embedding['representations'][33]
    #     return recombine_embedding
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        _, _, rec_id, attP_str, attB_str, rec_str, label = self.data.iloc[index, :]
        attachment_site_int = torch.from_numpy(label_attachment_site(attB_str, attP_str, nucleotide_to_number, self.attachment_site_max_len))
        # recombinases_int = self.rec_str_to_int(rec_id)
        recombinases_int = torch.from_numpy(label_attachment_site_protein(rec_str, CHARPROTSET, self.attachment_site_max_len))
        return attachment_site_int, recombinases_int, label

'''
############################################################################
                        model structure
############################################################################
'''
class MCANet(nn.Module):
    def __init__(self, 
                 hp: hyperparameter,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super(MCANet, self).__init__()
        self.rec_dim = hp.char_dim #1280
        self.att_dim = hp.att_dim  #64
        self.conv = hp.conv        #128
        self.drug_MAX_LENGTH = drug_MAX_LENGH #100
        self.drug_kernel = hp.drug_kernel #[4, 6, 8]
        self.protein_MAX_LENGTH = protein_MAX_LENGH #1000
        self.protein_kernel = hp.protein_kernel     #[4, 8, 12]
        self.drug_vocab_size = 5
        self.protein_vocab_size = 25
        self.attention_dim = hp.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
            self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = 29 #这个根据尺寸来决定
        # self.protein_MAX_LENGTH - \
        #     self.protein_kernel[0] - self.protein_kernel[1] - \
        #     self.protein_kernel[2] + 3
        self.drug_attention_head = 5
        self.protein_attention_head = 7
        self.mix_attention_head = hp.mix_attention_head #8

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.att_dim, padding_idx=0) # 5 -> 64
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.att_dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.att_dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.att_dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )

        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

        self.mix_attention_layer = nn.MultiheadAttention(
            self.attention_dim, self.mix_attention_head)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, protein): #durg:[256, 100],protein:[256, 50]
        # [B, F_O] -> [B, F_O, D_E]
        # [B, T_O] -> [B, T_O, D_E]
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        # [B, F_O, D_E] -> [B, D_E, F_O]
        # [B, T_O, D_E] -> [B, D_E, T_O]
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # [B, D_C, F_C] -> [F_C, B, D_C]
        # [B, D_C, T_C] -> [T_C, B, D_C]
        drug_QKV = drugConv.permute(2, 0, 1)
        protein_QKV = proteinConv.permute(2, 0, 1)

        # cross Attention
        # [F_C, B, D_C] -> [F_C, B, D_C]
        # [T_C, B, D_C] -> [T_C, B, D_C]
        drug_att, _ = self.mix_attention_layer(drug_QKV, protein_QKV, protein_QKV)
        protein_att, _ = self.mix_attention_layer(protein_QKV, drug_QKV, drug_QKV)

        # [F_C, B, D_C] -> [B, D_C, F_C]
        # [T_C, B, D_C] -> [B, D_C, T_C]
        drug_att = drug_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)

        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        # print('drugConv',drugConv.shape)
        # print('proteinConv',proteinConv.shape)
        
        
        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict


class Embedding(nn.Module): #将索引列表表示成tok+pos：[batch, seq_len+2, d_model]
    def __init__(self, hp: hyperparameter):
        super(Embedding, self).__init__()
        self.drug_vocab_size = 5
        self.att_dim = hp.att_dim
        self.drug_MAX_LENGH=100

        self.tok_embed = nn.Embedding(self.drug_vocab_size, self.att_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(self.drug_MAX_LENGH, self.att_dim)
        self.norm = nn.LayerNorm(self.att_dim)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len] #[64,43]
        pos = torch.arange(self.drug_MAX_LENGH,device=self.DEVICE, dtype=torch.long) # [seq_len]
        # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        #     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        #     36, 37, 38, 39, 40, 41, 42])
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len] #[64,43]
        # tensor([[ 0,  1,  2,  ..., 40, 41, 42],
        #         [ 0,  1,  2,  ..., 40, 41, 42],
        #         [ 0,  1,  2,  ..., 40, 41, 42],
        #         ...,
        #         [ 0,  1,  2,  ..., 40, 41, 42],
        #         [ 0,  1,  2,  ..., 40, 41, 42],
        #         [ 0,  1,  2,  ..., 40, 41, 42]])
        embedding = self.pos_embed(pos)           # [64, 43, 32]
        embedding = embedding + self.tok_embed(x) # [64, 43, 32] + [64, 43, 32]
        embedding = self.norm(embedding)
        return embedding

class Adapt_MCANet(nn.Module):
    def __init__(self, 
                 hp: hyperparameter,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super(Adapt_MCANet, self).__init__()
        self.rec_dim = hp.char_dim
        self.att_dim = hp.att_dim
        self.conv = hp.conv
        self.drug_MAX_LENGTH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGTH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = 5
        self.protein_vocab_size = 22
        self.attention_dim = hp.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
            self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
            self.protein_kernel[0] - self.protein_kernel[1] - \
            self.protein_kernel[2] + 3
        self.drug_attention_head = 5
        self.protein_attention_head = 7
        self.mix_attention_head = hp.mix_attention_head

        self.embedding = Embedding(hp)
        # self.drug_embed = nn.Embedding(
        #     self.drug_vocab_size, self.att_dim, padding_idx=0) # 5 -> 64
        #self.protein_embed = nn.Embedding(
            #self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.att_dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.rec_dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )

        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

        self.mix_attention_layer = nn.MultiheadAttention(
            self.attention_dim, self.mix_attention_head)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, proteinembed):
        # [B, F_O] -> [B, F_O, D_E]
        # [B, T_O] -> [B, T_O, D_E]
        # drugembed = self.drug_embed(drug)
        drugembed = self.embedding(drug)
        #proteinembed = self.protein_embed(protein)
        # [B, F_O, D_E] -> [B, D_E, F_O]
        # [B, T_O, D_E] -> [B, D_E, T_O]
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # [B, D_C, F_C] -> [F_C, B, D_C]
        # [B, D_C, T_C] -> [T_C, B, D_C]
        drug_QKV = drugConv.permute(2, 0, 1)
        protein_QKV = proteinConv.permute(2, 0, 1)

        # cross Attention
        # [F_C, B, D_C] -> [F_C, B, D_C]
        # [T_C, B, D_C] -> [T_C, B, D_C]
        drug_att, _ = self.mix_attention_layer(drug_QKV, protein_QKV, protein_QKV)
        protein_att, _ = self.mix_attention_layer(protein_QKV, drug_QKV, drug_QKV)

        # [F_C, B, D_C] -> [B, D_C, F_C]
        # [T_C, B, D_C] -> [B, D_C, T_C]
        drug_att = drug_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)

        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict

class MCANet_noshare(nn.Module):
    def __init__(self, 
                 hp: hyperparameter,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super(MCANet_noshare, self).__init__()
        self.rec_dim = hp.char_dim
        self.att_dim = hp.att_dim
        self.conv = hp.conv
        self.drug_MAX_LENGTH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGTH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = 5
        self.protein_vocab_size = 22
        self.attention_dim = hp.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
            self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
            self.protein_kernel[0] - self.protein_kernel[1] - \
            self.protein_kernel[2] + 3
        self.drug_attention_head = 5
        self.protein_attention_head = 7
        self.mix_attention_head = hp.mix_attention_head

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.att_dim, padding_idx=0)
        #self.protein_embed = nn.Embedding(
            #self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.att_dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.rec_dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )

        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

        self.mix_attention_layer = nn.MultiheadAttention(
            self.attention_dim, self.mix_attention_head)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, proteinembed):
        # [B, F_O] -> [B, F_O, D_E]
        # [B, T_O] -> [B, T_O, D_E]
        drugembed = self.drug_embed(drug)
        #proteinembed = self.protein_embed(protein)
        # [B, F_O, D_E] -> [B, D_E, F_O]
        # [B, T_O, D_E] -> [B, D_E, T_O]
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # [B, D_C, F_C] -> [F_C, B, D_C]
        # [B, D_C, T_C] -> [T_C, B, D_C]
        drug_QKV = drugConv.permute(2, 0, 1)
        protein_QKV = proteinConv.permute(2, 0, 1)

        # cross Attention
        # [F_C, B, D_C] -> [F_C, B, D_C]
        # [T_C, B, D_C] -> [T_C, B, D_C]
        drug_att, _ = self.mix_attention_layer(drug_QKV, drug_QKV, drug_QKV)
        protein_att, _ = self.mix_attention_layer(protein_QKV, protein_QKV, protein_QKV)

        # [F_C, B, D_C] -> [B, D_C, F_C]
        # [T_C, B, D_C] -> [B, D_C, T_C]
        drug_att = drug_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)

        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict



'''
############################################################################
                       new model structure
############################################################################
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.nn.utils.weight_norm import weight_norm


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class DrugBAN(nn.Module):
    def __init__(self, **config):
        super(DrugBAN, self).__init__()
        drug_in_feats = 128 #config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = 1280 #config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = [128, 128, 128] #config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = 1280 #config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = [128, 128, 128] # config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = [3, 3, 3]#[3, 6, 9] #config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = 256 #config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = 512 #config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = 128 #config["DECODER"]["OUT_DIM"]
        drug_padding = True #config["DRUG"]["PADDING"]
        protein_padding = True# config["PROTEIN"]["PADDING"]
        out_binary = 2 #1 #config["DECODER"]["BINARY"]
        ban_heads = 2 # config["BCN"]["HEADS"]
        #self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
        #                                    padding=drug_padding,
        #                                    hidden_feats=drug_hidden_feats)
        
        self.DNA_extractor = DNACNN(drug_embedding, num_filters, kernel_size, protein_padding)
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):#drug, proteinembed
        v_d = self.DNA_extractor(bg_d)
        v_p = self.protein_extractor(v_p)
        f, att = self.bcn(v_d, v_p)
        score = self.mlp_classifier(f)
        if mode == "train":
            return score#v_d, v_p, f, score
        elif mode == "eval":
            return score#v_d, v_p, score, att

class DNACNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(DNACNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(5, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(5, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        # if padding:
        #     self.embedding = nn.Embedding(700, embedding_dim, padding_idx=0)
        # else:
        #     self.embedding = nn.Embedding(700, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        # v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=2):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]



class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q ###
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)




'''
############################################################################
                        loss function
############################################################################
'''
class PolyLoss(nn.Module):
    def __init__(self, weight_loss, batch_size, DEVICE, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_loss, reduction='none')
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        one_hot = torch.zeros((self.batch_size, 2), device=self.DEVICE).scatter_(1, torch.unsqueeze(labels, dim=-1), 1)
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        ce = self.CELoss(predicted, labels)
        poly1 = ce + self.epsilon * (1-pt)
        return torch.mean(poly1)


class CELoss(nn.Module):
    def __init__(self, weight_CE, DEVICE):
        super(CELoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_CE)
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        return self.CELoss(predicted, labels)


#------------------>>>
def get_entropy(probs):
    ent = -(probs.mean(0) * torch.log2(probs.mean(0) + 1e-12)).sum(0, keepdim=True)
    # A.mean(0) 计算每一列的平均值
    return ent

def get_cond_entropy(probs):
    cond_ent = -(probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
    return cond_ent

class Tim_loss(nn.Module):
    def __init__(self):
        super(Tim_loss, self).__init__()
        
        self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, logits, label):
        loss = self.criterion_loss(logits.view(-1, config.num_class), label.view(-1)) #交叉熵结果
        loss = (loss.float()).mean() #求平均
        loss = (loss - config.b).abs() + config.b      #b = 0.06
        
        # Q_sum = len(logits)
        logits = F.softmax(logits, dim=1)  # softmax归一化

        sum_loss = loss + get_entropy(logits) - get_cond_entropy(logits)
        return sum_loss[0]


class FocalLoss(nn.Module):
    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            alpha = torch.tensor([alpha, 1 - alpha]) #额外添加
            if isinstance(alpha, Variable): #isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0) 
        #new的用途：
        #创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容。
        #可以指定新创建的张量的形状，并且赋予随机值,这里用0填充

        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)  #做独热码 #https://www.cnblogs.com/qllouhenglei/p/13590089.html

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean() #
        else:
            loss = batch_loss.sum()
        return loss


'''
############################################################################
                    Early Stopping Class
############################################################################
'''
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, savepath=None, patience=7, verbose=False, delta=0, num_n_fold=0, log_fn=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False
        self.delta = delta
        self.num_n_fold = num_n_fold
        self.savepath = savepath
        self.log_fn = log_fn

    def __call__(self, score, model, num_epoch):

        if self.best_score == -np.inf:
            self.save_checkpoint(score, model, num_epoch)
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.log_fn(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model, num_epoch)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model, num_epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.log_fn(
                f'Have a new best checkpoint: ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.savepath +
                '/valid_best_checkpoint.pth')



'''
############################################################################
                    validation metrics visulization
############################################################################
'''
def test_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):
    if isinstance(MODEL, list):
        for item in MODEL:
            item.eval()
    else:
        MODEL.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds, proteins, labels = data
            compounds = compounds.to(DEVICE)
            proteins = proteins.to(DEVICE)
            labels = labels.to(DEVICE)

            if isinstance(MODEL, list):
                predicted_scores = torch.zeros(2).to(DEVICE)
                for i in range(len(MODEL)):
                    predicted_scores = predicted_scores + \
                        MODEL[i](compounds, proteins)
                predicted_scores = predicted_scores / FOLD_NUM
            else:
                predicted_scores = MODEL(compounds, proteins)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            # predicted_scores = torch.sigmoid(predicted_scores).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    return Y, P, test_loss, Accuracy, Precision, Recall, AUC, PRC


def test_model(MODEL, dataset_loader, save_path, LOSS, DEVICE, dataset_class="Train", save=True, FOLD_NUM=1, log_fn=print):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_loader)),
        total=len(dataset_loader),
        ncols=100)
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_precess(
        MODEL, test_pbar, LOSS, DEVICE, FOLD_NUM)
    if save:
        if FOLD_NUM == 1:
            filepath = save_path + \
                "/{}_prediction.txt".format(dataset_class)
        else:
            filepath = save_path + \
                "/{}_ensemble_prediction.txt".format(dataset_class)
        with open(filepath, 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(dataset_class, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    log_fn(results)
    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test

def show_result(save_path, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List, Ensemble=False, log_fn=print):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(
        Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)

    if Ensemble == False:
        log_fn("The model's results:")
        filepath = "{}/results.txt".format(save_path)
    else:
        log_fn("The ensemble model's results:")
        filepath = "{}/ensemble_results.txt".format(save_path)
    with open(filepath, 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(
            Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(
            Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(
            Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')
    log_fn('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    log_fn('Precision(std):{:.4f}({:.4f})'.format(
        Precision_mean, Precision_var))
    log_fn('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    log_fn('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    log_fn('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))

'''
############################################################################
                        main functions
############################################################################
'''

def main(args: argsInput, DEVICE, SEED=114514, K_Fold=5, LOSS='PolyLoss'):
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # copy script to out directory
    file_abspath = os.path.abspath(__file__)
    running_time = get_datetime()
    args.out_dir = os.path.join(args.out_dir, running_time)
    code_dir = os.path.join(args.out_dir, "code")
    if os.path.exists(code_dir):
        shutil.rmtree(code_dir) # 递归删除文件
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy(file_abspath, code_dir)

    makeLog = MakeLogClass("{}/log.txt".format(args.out_dir)).make
    makeLog("copy script done!")

    '''make a log file'''

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load testData'''
    makeLog("load Testdata")
    makeLog("dataset: {}\nshuffle_times: {}\n".format(args.datasets_path, args.shuffle_times))
    testData = pd.read_csv("{}/PASTE_{}_test.txt".format(args.datasets_path, args.shuffle_times), sep='\t', header=None)
    makeLog("load finished")

    '''metrics'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    for i_fold in [0]: #range(K_Fold)[0]: #先跑第一折看看结果,5折中的第一折
        ''' load dataset from text file '''
        i_fold =0
        makeLog("\n{}-fold:".format(i_fold))
        trainData, validData = get_k_fold_datasets(args, fold_index=i_fold, k_fold=K_Fold)
        train_dataset = CustomDataset(data=trainData, 
                                      embedding_dir=args.embedding_dir, 
                                      attachment_site_max_len=hp.attachment_site_max_len)

        valid_dataset = CustomDataset(data=validData, 
                                      embedding_dir=args.embedding_dir,
                                      attachment_site_max_len=hp.attachment_site_max_len)

        test_dataset = CustomDataset(data=testData, 
                                     embedding_dir=args.embedding_dir,
                                     attachment_site_max_len=hp.attachment_site_max_len)
                                    
        train_size = len(train_dataset)
        
        makeLog('Number of Train set: {}'.format(len(train_dataset)))
        makeLog('Number of Valid set: {}'.format(len(valid_dataset)))
        makeLog('Number of Test set: {}'.format(len(test_dataset)))

        train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=hp.num_workers, drop_last=True)
        valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=hp.num_workers, drop_last=True)
        test_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=hp.num_workers, drop_last=True)

        '''create model'''
        model = MCANet(hp, protein_MAX_LENGH=hp.protein_max_len, drug_MAX_LENGH=hp.attachment_site_max_len * 2).to(DEVICE)
        # model = MCANet_noshare(hp, protein_MAX_LENGH=hp.protein_max_len, drug_MAX_LENGH=hp.attachment_site_max_len * 2).to(DEVICE)
        # model = Adapt_MCANet(hp, protein_MAX_LENGH=hp.protein_max_len, drug_MAX_LENGH=hp.attachment_site_max_len * 2).to(DEVICE)
        # model = DrugBAN().to(DEVICE)

        """Initialize weights"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        """create optimizer and scheduler"""
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)
        if LOSS == 'PolyLoss':
            Loss = PolyLoss(weight_loss=None, batch_size=hp.Batch_size,
                            DEVICE=DEVICE, epsilon=hp.loss_epsilon)
        
        elif LOSS =='FocalLoss':
            Loss = FocalLoss()
        else:
            Loss = CELoss(weight_CE=None, DEVICE=DEVICE)

        """Output files"""
        save_path = args.out_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        early_stopping = EarlyStopping(savepath=save_path, patience=hp.Patience, verbose=True, delta=0, log_fn=makeLog)

        """Start training."""
        makeLog('Training...')
        for epoch in range(1, hp.Epoch + 1):
            if early_stopping.early_stop == True:
                break
            train_pbar = tqdm(
                enumerate(BackgroundGenerator(train_dataset_loader)),
                total=len(train_dataset_loader),
                ncols=100
            )

            '''train'''
            train_losses_in_epoch = []
            model.train()
            for train_i, train_data in train_pbar:
                train_compounds, train_proteins, train_labels = train_data
                train_compounds = train_compounds.to(DEVICE)
                train_proteins = train_proteins.to(DEVICE)
                train_labels = train_labels.to(DEVICE)

                optimizer.zero_grad()

                predicted_interaction = model(train_compounds, train_proteins)
                train_loss = Loss(predicted_interaction, train_labels)
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss

            valid_pbar = tqdm(
                enumerate(BackgroundGenerator(valid_dataset_loader)),
                total=len(valid_dataset_loader),
                ncols=100
            )

            """valid"""
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:

                    valid_compounds, valid_proteins, valid_labels = valid_data

                    valid_compounds = valid_compounds.to(DEVICE)
                    valid_proteins = valid_proteins.to(DEVICE)
                    valid_labels = valid_labels.to(DEVICE)

                    valid_scores = model(valid_compounds, valid_proteins)
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_losses_in_epoch.append(valid_loss.item())
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                    # valid_scores = torch.sigmoid(valid_scores).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]

                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)
    
            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                            f'train_loss: {train_loss_a_epoch:.5f} ' +
                            f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                            f'valid_AUC: {AUC_dev:.5f} ' +
                            f'valid_PRC: {PRC_dev:.5f} ' +
                            f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                            f'valid_Precision: {Precision_dev:.5f} ' +
                            f'valid_Reacll: {Reacll_dev:.5f} ')
            makeLog(print_msg)

            '''save checkpoint and make decision when early stop'''
            early_stopping(AUC_dev, model, epoch)

        '''load best checkpoint'''
        model.load_state_dict(torch.load(early_stopping.savepath + '/valid_best_checkpoint.pth'))

        '''test model'''
        trainset_test_stable_results, _, _, _, _, _ = test_model(
            model, train_dataset_loader, save_path, Loss, DEVICE, dataset_class="Train", FOLD_NUM=1, log_fn=makeLog)
        validset_test_stable_results, _, _, _, _, _ = test_model(
            model, valid_dataset_loader, save_path, Loss, DEVICE, dataset_class="Valid", FOLD_NUM=1, log_fn=makeLog)
        testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_loader, save_path, Loss, DEVICE, dataset_class="Test", FOLD_NUM=1, log_fn=makeLog)
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)
        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')

    show_result(save_path, Accuracy_List_stable, Precision_List_stable, Recall_List_stable, AUC_List_stable, AUPR_List_stable, Ensemble=False)

if __name__ == "__main__":
    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--datasets-path", type=str)
    #parser.add_argument("--embedding-dir", type=str)
    #parser.add_argument("--shuffle-times", type=int)
    #parser.add_argument("--identity", type=str)
    #parser.add_argument("--out-dir", type=str)
    #args = parser.parse_args()

    args = argsInput()
    this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args, this_device)


'''
python /tangzehua/tangzehua/RecCrossNet/code/model_1.py

'''
