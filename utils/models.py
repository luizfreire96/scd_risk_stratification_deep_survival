from torch import nn
import torch.nn.init as init
import torch
from functools import reduce
from operator import __add__
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable, List
from torch import Tensor


class CNN_LSTM_net(nn.Module):
    def __init__(self, input_channels=3, seq_len=1196000, n_additional_features=0, 
                 stride=16, kernel_size=25, dilation=[5, 5, 5], lstm_hidden_size=128):
        super().__init__()
        
        # Camada 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=kernel_size, 
                      stride=stride, dilation=dilation[0]),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=10, stride=6)
        )

        # Camada 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=kernel_size//2, stride=stride//2, dilation=dilation[1]),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=7, stride=4)
        )
        
        # Camada 3
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size//3, stride=stride//3, dilation=dilation[2]),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=5, stride=3),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True)
        
        # Camadas Fully Connected
        self.fc_additional = nn.Sequential(
            nn.Linear(lstm_hidden_size + n_additional_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )
        self.n_additional_features = n_additional_features


    def forward(self, x, x_tab=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        if self.n_additional_features > 0 and x_tab is not None:
            x_final = torch.cat((last_output, x_tab), dim=1)
            return self.fc_additional(x_final)
        else:
            return self.fc(last_output)
