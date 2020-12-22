import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import shutil
import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.data
import torch.optim as optim
import random
import time

cuda=torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

class ECGDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        self.data = pd.read_csv(path, header=None)

    def __getitem__(self, idx):
        x = self.data.loc[idx, :186].values  # removed the label
        return x

    def __len__(self):
        return len(self.data)

class AE_opt:
    batch_size=64
    workers=2
    lr=0.001
    # normal_train_path=root_path+"/normal_train.csv"
    # normal_valid_path=root_path+"/normal_valid.csv"
    # patient_valid_path=root_path+"/patient_valid.csv"


class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # input=(seq_len,batch_input,input_size)(187,64,1)
        # lstm=(input_size,hidden_size)
        self.hidden_size = 20
        self.lstm1 = nn.LSTMCell(1, self.hidden_size)
        self.lstm2 = nn.LSTMCell(1, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def Encoder(self, inp):

        ht = torch.zeros(inp.size(0), self.hidden_size, dtype=torch.float, device=device)
        ct = torch.zeros(inp.size(0), self.hidden_size, dtype=torch.float, device=device)

        for input_t in inp.chunk(inp.size(1), dim=1):
            ht, ct = self.lstm1(input_t, (ht, ct))

        return ht, ct

    def Decoder(self, ht, ct):

        ot = torch.zeros(ht.size(0), 1, dtype=torch.float, device=device)
        outputs = torch.zeros(ht.size(0), 187, dtype=torch.float, device=device)

        for i in range(187):
            ht, ct = self.lstm2(ot, (ht, ct))
            ot = self.sigmoid(self.linear(ht))
            outputs[:, i] = ot.squeeze()

        return outputs

    def forward(self, inp):

        he, ce = self.Encoder(inp)  # hidden encoder,cell_state encoder
        out = self.Decoder(he, ce)
        return torch.flip(out, dims=[1]), torch.log(F.softmax(he, dim=1))