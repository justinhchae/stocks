# adapted from
# https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca

import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sequence_length = 59


def make_sequence(data, data_col, seq_len=sequence_length):
    """
    return sequenced stock data as torch tensors
    of shape len(data) by sequence length
    """
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
                 , input_dim=1
                 , hidden_dim=sequence_length
                 , output_dim=1
                 # , num_layers=2
                 ):

        super(Model, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(1,1, self.hidden_dim),
                torch.zeros(1,1, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        # lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        lstm_out, self.hidden = self.lstm(input.view(len(input), 1, -1), self.hidden)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        # y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        # return y_pred.view(-1)

        predictions = self.linear(lstm_out.view(len(input), -1))
        print(predictions)
        return predictions[-1]


def train_model_1(df, epochs=3, learning_rate=0.01):

    data = prep_arr(df, time_col='t', data_col='c')

    x_train, y_train = make_sequence(data=data, data_col='c')

    model = Model()

    loss_function = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start loop step 1
    model.zero_grad()

    model.hidden = model.init_hidden()

    y_pred = model(x_train[0])
    print(y_pred)
