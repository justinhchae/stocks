# adapted from
# https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca

import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import numpy as np

def make_sequence(data, data_col, seq_len=100):
    x, y = [], []
    arr = data[data_col]

    for i in range(len(arr) - seq_len):
        x_i = arr[i : i + seq_len]
        y_i = arr[i + 1 : i + seq_len + 1]

        x.append(x_i)
        y.append(y_i)

    x_arr = np.array(x).reshape(-1, seq_len)
    y_arr = np.array(y).reshape(-1, seq_len)

    x_var = Variable(torch.from_numpy(x_arr).float())
    y_var = Variable(torch.from_numpy(y_arr).float())

    return x_var, y_var

def prep_arr(df, time_col, data_col):
    data_dict = {}

    time_index = df[time_col].values
    data_values = df[data_col].values

    data_dict.update({time_col:time_index})
    data_dict.update({data_col:data_values})

    return data_dict

class Model(nn.Module):
    def __init__(self
                 , input_size = 1 # n features in the input, one stock price, one input
                 , output_size = 1 # one in, one out
                 , hidden_size = 60 # 60 minutes in an hour (but mostly aribtrary)
                 , number_of_layers=2
                 , max_norm=0.001
                 # , dropout_probability=0.1
                 , batch_size=64
                 , sequence_length=100
                 , learning_rate=0.005
                 , max_init_param=0.01
                 , device="cpu"
                 , sequence_step_size=None
                 , learning_rate_decay=.8
                 ):

        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.number_of_layers = number_of_layers
        self.max_norm = max_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_init_param = max_init_param
        self.learning_rate_decay = learning_rate_decay

        if sequence_step_size is None:
            self.sequence_step_size = sequence_length
        else:
            self.sequence_step_size = sequence_step_size

        if device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device =  torch.device("cpu")

        rnns = [nn.LSTM(self.input_size, self.hidden_size) for _ in range(number_of_layers)]
        self.rnns = nn.ModuleList(rnns)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, input, states):
        X = self.dropout(input)
        for i, rnn in enumerate(self.rnns):
            X, states[i] = rnn(X, states[i])
            X = self.dropout(X)

        output = self.fc(X)

        return output, states


def train_model_1(df):
    data = prep_arr(df, time_col='t', data_col='c')

    x_train, y_train = make_sequence(data=data, data_col='c', seq_len=100)

    model = Model()

    model.train()

    # simulate a single loop

    model.zero_grad()
    



