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


'''
############################################################################
                            hyperPamameters
############################################################################
'''
class argsInput: #指定路径，主要是关于数据
    def __init__(self) -> None:
        self.shuffle_times = 1
        self.identity = 0.8
        self.root_dir = "/lizutan/code/MCANet"
        self.datasets_path = "{}/dataset/pipeline_7/identity_{}_shuffle_{}".format(self.root_dir, self.identity, self.shuffle_times)

        #编码路径
        self.protein_embedding_dir = "{}/preprocess/esm_embedding_data/protein_700".format(self.root_dir)
        self.dna_embedding_dir = "{}/preprocess/dnabert_embedding/pipeline_7/identity_0.8_shuffle_1".format(self.root_dir)
            
        self.out_dir =self.root_dir +  "/result/identity_{}_shuffle_{}".format(self.identity, self.shuffle_times)

# /lizutan/code/MCANet/preprocess/esm_embedding_data
# /lizutan/code/MCANet/preprocess/esm_embedding_data/protein_700/recombinases_4804.pt
# class argsInput:
#     def __init__(self) -> None:
#         self.shuffle_times = 1
#         self.identity = 0.8
#         self.root_dir ='/lizutan/code/MCANet' #"/tangzehua/tangzehua/RecCrossNet"
#         self.datasets_path = "{}/dataset/pipeline_5/identity_{}_shuffle_{}".format(self.root_dir, self.identity, self.shuffle_times)
#         self.embedding_dir = "{}/preprocess/esm_embedding_data/protein_700".format(self.root_dir)
#         self.out_dir = "/lizutan/code/MCANet/result/DrugBAN_model_identity_{}_shuffle_{}".format(self.identity, self.shuffle_times)

class hyperparameter:
    def __init__(self):
        self.Learning_rate = 1e-4
        self.Epoch = 200
        self.Batch_size = 256
        self.Patience = 50
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.embed_dim = 64
        self.protein_kernel = [3, 3, 3]#[4, 8, 12]
        self.drug_kernel = [3, 3, 3]#[4, 6, 8]
        self.conv = 40
        self.protein_emb_dim = 1280 # esm1b
        self.dna_emb_dim = 768      # dnabert
        self.loss_epsilon = 1
        self.num_workers = 16
        self.protein_max_len = 700
        self.attachment_site_max_len = 47 #dnabert编码完是47

nucleotide_to_number = {
    "A": 1, "C": 2, "G": 3, "T": 4
}

'''
############################################################################
                            Helper Functions
############################################################################
'''
def get_datetime(): #获取时间
    '''output: month_day_hour_minute'''
    current_datetime = datetime.datetime.now()
    month = current_datetime.month
    day = current_datetime.day
    hour = current_datetime.hour
    minute = current_datetime.minute
    return "{}_{}_{}_{}".format(month, day, hour, minute)

def label_attachment_site(attB, attP, smi_ch_ind, MAX_HALF_SITE_LEN): #在embedding的时候没用到这一部分
    X = np.zeros(MAX_HALF_SITE_LEN * 2, np.int64())
    for i, ch in enumerate(attB[:MAX_HALF_SITE_LEN]):
        if ch in smi_ch_ind:
            X[i] = smi_ch_ind[ch]
    for i, ch in enumerate(attP[:MAX_HALF_SITE_LEN]):
        if ch in smi_ch_ind:
            X[i+MAX_HALF_SITE_LEN] = smi_ch_ind[ch]
    return X

class MakeLogClass:
    def __init__(self, log_file):
        self.log_file = log_file
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
#
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
rec_embedding = torch.load('/lizutan/code/MCANet/preprocess/esm_embedding_data/protein_700/recombinases_24719.pt')
attP_embed = np.load("/lizutan/code/MCANet/preprocess/dnabert_embedding/pipeline_7/identity_0.8_shuffle_1/attB/attB_0.npz")
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
    def __init__(self, data, args: argsInput, attachment_site_max_len=50): #初始化
        self.data = data #读入的数据，datafrmae格式
        self.attachment_site_max_len = attachment_site_max_len
        self.protein_embed_dir = args.protein_embedding_dir
        self.dna_embed_dir = args.dna_embedding_dir
    
    #重组酶读取
    def rec_str_to_int(self, recombinases_id): #读取esm-1b文件 #LSR
        rec_embedding = torch.load("{}/{}.pt".format(self.protein_embed_dir, recombinases_id))
        recombine_embedding = rec_embedding['representations'][33]
        return recombine_embedding
    '''
    # rec_embedding #torch.Size([700, 1280])
    {'label': 'recombinases_24719', 'representations': {33: tensor([[ 0.0486, -0.1166,  0.0347,  ...,  0.0378,  0.0892,  0.1587],
        [ 0.0640, -0.0445,  0.0354,  ..., -0.0693,  0.1725,  0.0071],
        [ 0.0496,  0.1803, -0.2004,  ..., -0.0690,  0.2289,  0.1827],
        ...,
        [ 0.0237,  0.0762,  0.0498,  ..., -0.0545,  0.0305,  0.0163],
        [-0.0307,  0.0880,  0.0231,  ..., -0.0190,  0.0273, -0.0348],
        [ 0.0161,  0.0792,  0.0347,  ..., -0.0355,  0.0934, -0.0639]])}}
    '''

    # attp和attb读取
    def site_id_to_int(self, attP_id, attB_id):#读取dnabert文件
        attP_embed = np.load("{}/attP/{}.npz".format(self.dna_embed_dir, attP_id))
        attB_embed = np.load("{}/attB/{}.npz".format(self.dna_embed_dir, attB_id))
        assert attP_id == str(attP_embed['index'])
        assert attB_id == str(attB_embed['index'])
        attachment_site_embed = np.concatenate([attB_embed['representations'], attP_embed['representations']], axis=0)
        return attachment_site_embed
    
    '''
    >>> attP_embed['index']
    array('attB_0', dtype='<U6')
    >>> attP_embed['representations'] #(47, 768)
    array([[-0.5547674 ,  1.7012819 ,  0.94594115, ..., -1.2335695 ,0.18116936,  0.8817105 ],
        [-0.37259725,  0.50960326,  0.23636307, ..., -1.2921443 ,0.80167764,  1.2283553 ],
        [-1.2407184 ,  2.3635552 ,  1.5922157 , ..., -1.2617459 ,
    '''

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        attP_id, attB_id, rec_id, attP_str, attB_str, rec_str, label = self.data.iloc[index, :] #pandas索引
        attachment_site_int = torch.from_numpy(self.site_id_to_int(attP_id, attB_id))
        recombinases_int = self.rec_str_to_int(rec_id)
        return attachment_site_int, recombinases_int, label

# the using of function
# train_dataset = CustomDataset(data=trainData, args= args, attachment_site_max_len=hp.attachment_site_max_len)
# 返回的是 attachment_site_int, recombinases_int, label

'''
############################################################################
                        model structure
############################################################################
'''
class MCANet(nn.Module):
    def __init__(self, hp: hyperparameter,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100,
                 log_fn=print):
        super(MCANet, self).__init__()
        self.dna_emb_dim = hp.dna_emb_dim
        self.rec_emb_dim = hp.protein_emb_dim
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
        self.mix_attention_head = 5
        self.log_fn = log_fn
        #
        #self.drug_embed = nn.Embedding(
            #self.drug_vocab_size, self.dim, padding_idx=0)
        #self.protein_embed = nn.Embedding(
            #self.protein_vocab_size, self.dim, padding_idx=0)
        #
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dna_emb_dim, out_channels=self.conv,
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
            nn.Conv1d(in_channels=self.rec_emb_dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        #
        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)
        #
        self.mix_attention_layer = nn.MultiheadAttention(
            self.attention_dim, self.mix_attention_head)
        #
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)
        #
    def forward(self, drugembed, proteinembed):
        # drugembe torch.Size([256, 94, 768])
        # proteinembed torch.Size([256, 700, 1280])
        #
        #drugembed = self.drug_embed(drug)
        #proteinembed = self.protein_embed(protein)
        #
        drugembed = drugembed.permute(0, 2, 1) #[bs,768,47]
        proteinembed = proteinembed.permute(0, 2, 1) #[bs,1280,700]
        #
        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed) #drugConv torch.Size([256, 768, 94])
        proteinConv = self.Protein_CNNs(proteinembed) #proteinConv torch.Size([256, 160, 679])
        # [B, D_C, F_C] -> [F_C, B, D_C]
        # [B, D_C, T_C] -> [T_C, B, D_C]
        drug_QKV = drugConv.permute(2, 0, 1) #[94 ,256, 768]
        protein_QKV = proteinConv.permute(2, 0, 1)#[679, 256, 160]
        #
        # cross Attention
        # [F_C, B, D_C] -> [F_C, B, D_C]
        # [T_C, B, D_C] -> [T_C, B, D_C]
        drug_att, _ = self.mix_attention_layer(drug_QKV, protein_QKV, protein_QKV) #drug_att torch.Size([79, 256, 160])
        protein_att, _ = self.mix_attention_layer(protein_QKV, drug_QKV, drug_QKV) #protein_att torch.Size([679, 256, 160])
        # [F_C, B, D_C] -> [B, D_C, F_C]
        # [T_C, B, D_C] -> [B, D_C, T_C]
        drug_att = drug_att.permute(1, 2, 0) #drug_att torch.Size([256, 160, 79])
        protein_att = protein_att.permute(1, 2, 0) #protein_att torch.Size([256, 160, 679])
        #
        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5
        #
        #抹掉第2个纬度的数据，用最大值替代，这一步需要考虑
        drugConv = self.Drug_max_pool(drugConv).squeeze(2) #drugConv torch.Size([256, 160])
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2) #proteinConv torch.Size([256, 160])
        #
        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3) #输出是不是可以有一个softmax或者sigmoid，替换后面的这个
        return predict

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
        # scatter_标签转换为one-hot编码
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

            if isinstance(MODEL, list): #多个模型列表
                predicted_scores = torch.zeros(2).to(DEVICE) #tensor([0., 0.])
                for i in range(len(MODEL)):
                    predicted_scores = predicted_scores + MODEL[i](compounds, proteins)
                predicted_scores = predicted_scores / FOLD_NUM
            else:
                predicted_scores = MODEL(compounds, proteins)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()                           #真实标签
            # predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy() #softmax用于二分类
            predicted_scores = torch.sigmoid(predicted_scores).to('cpu').data.numpy() #softmax用于二分类
            
            predicted_labels = np.argmax(predicted_scores, axis=1)                   #预测的标签-label
            predicted_scores = predicted_scores[:, 1]                                #预测得分，第二个纬度上的一个值 

            Y.extend(correct_labels)  #真实标签
            P.extend(predicted_labels)#预测的标签-label
            S.extend(predicted_scores)#预测得分，第二个纬度上的一个值 
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
    test_pbar = tqdm(enumerate(BackgroundGenerator(dataset_loader)),
        total=len(dataset_loader),
        ncols=100) #[index,(compounds, proteins, labels)]
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
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
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
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')

    log_fn('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    log_fn('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
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
        shutil.rmtree(code_dir)
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

    for i_fold in range(K_Fold):
        ''' load dataset from text file '''
        makeLog("\n{}-fold:".format(i_fold))
        trainData, validData = get_k_fold_datasets(args, fold_index=i_fold, k_fold=K_Fold)
        train_dataset = CustomDataset(data=trainData, 
                                      args= args, 
                                      attachment_site_max_len=hp.attachment_site_max_len)

        valid_dataset = CustomDataset(data=validData, 
                                      args= args, 
                                      attachment_site_max_len=hp.attachment_site_max_len)

        test_dataset = CustomDataset(data=testData, 
                                     args= args, 
                                     attachment_site_max_len=hp.attachment_site_max_len)
                                    
        train_size = len(train_dataset)
        
        makeLog('Number of Train set: {}'.format(len(train_dataset)))
        makeLog('Number of Valid set: {}'.format(len(valid_dataset)))
        makeLog('Number of Test set: {}'.format(len(test_dataset)))

        train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=hp.num_workers, drop_last=True)
        valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=hp.num_workers, drop_last=True)
        test_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=hp.num_workers, drop_last=True)

        '''create model'''
        model = MCANet(hp, protein_MAX_LENGH=hp.protein_max_len, drug_MAX_LENGH=hp.attachment_site_max_len * 2, log_fn=makeLog).to(DEVICE)
        # model = MCANet(hp, protein_MAX_LENGH=hp.protein_max_len, drug_MAX_LENGH=hp.attachment_site_max_len * 2).to('cuda')
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
                    # valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                    valid_scores = torch.sigmoid(valid_scores).to('cpu').data.numpy()
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
python /tangzehua/tangzehua/RecCrossNet/code/model_5.py

'''
