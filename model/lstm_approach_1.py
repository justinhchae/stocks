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

try:
  print('Number of GPUs:',torch.cuda.device_count())
  print('GPU Card:', torch.cuda.get_device_name(0))
  device = torch.device('cuda:0')
except:
  print("No GPU this session, learning with CPU.")
  device = torch.device("cpu")


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
                 , device=device
                 ):

        super(Model, self).__init__()

        self.device=device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(1,1, self.hidden_dim, device=self.device),
                torch.zeros(1,1, self.hidden_dim, device=self.device))

    def forward(self, input):
        # Forward pass through LSTM layer

        lstm_out, self.hidden = self.lstm(input.view(len(input), 1, -1), self.hidden)
        # Only take the output from the final timetep

        predictions = self.linear(lstm_out.view(len(input), -1))
        return predictions[-1]


def train_model_1(df, epochs=3, learning_rate=0.01):

    #TODO: batching, nn layers, early stopping

    losses = []
    preds = []
    targets = []

    model = Model()
    model = model.to(model.device)
    model.train()

    data = prep_arr(df, time_col='t', data_col='c')
    x_train, y_train = make_sequence(data=data, data_col='c')

    x_train = x_train.to(model.device)
    y_train = y_train.to(model.device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start loop step 1
    for idx, input in enumerate(x_train):
        model.zero_grad()
        model.hidden = model.init_hidden()

        y_pred = model(input)
        preds.append(y_pred.item())

        y = torch.tensor([y_train[idx][0]])
        targets.append(y.item())

        # compare the last time step prediction to the first value of the next time step
        y_pred = y_pred.to(model.device)
        y = y.to(model.device)
        loss = loss_function(y_pred, y)

        losses.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if idx % 1000 == 0:
            print('Epoch: {}.............'.format(idx), end=' ')
            print("Loss: {:.4f}".format(loss.item()))

        # end loop 1

    plt.figure()
    plt.plot(losses, label='train loss')
    plt.title('loss graph lstm approach 1')
    plt.legend()
    plt.savefig('figures/lstm_approach_1_loss.png')
    plt.show()

    plt.figure()
    plt.plot(targets, label='targets')
    plt.plot(preds, label='predictions')
    plt.title('price prediction to target lstm approach 1')
    plt.legend()
    plt.savefig('figures/lstm_approach_1_predictions.png')
    plt.show()
