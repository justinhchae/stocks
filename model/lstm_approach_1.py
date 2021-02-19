# adapted from
# https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
  print('Number of GPUs:',torch.cuda.device_count())
  print('GPU Card:', torch.cuda.get_device_name(0))
  device = torch.device('cuda:0')
except:
  print("No GPU this session, learning with CPU.")
  device = torch.device("cpu")

class Data(Dataset):
    def __init__(self, data, window, yhat='c', step_size=1):
        self.data = data
        self.window = window
        self.step_size = step_size
        # self._xdata = # index of times
        self.yhat = data.columns.get_loc(yhat)
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        # TODO: only pass data columns, not time
        #FIXME: check slicing and target - prediction loss
        x = self.data.iloc[index: index + self.window, 1:]
        y = self.data.iloc[index + self.window, self.yhat:self.yhat+1]
        return torch.tensor(x.to_numpy(), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.window

    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def __getsize__(self):
        return (self.__len__())

class Model(nn.Module):

    def __init__(self
                 , input_dim
                 , seq_length
                 , hidden_dim=20
                 , output_dim=1
                 , num_layers=1
                 , device=device
                 ):

        super(Model, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * self.seq_length, output_dim)

    def init_hidden(self, batch_size):
        # This is what we'll initialise our hidden state as
        self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device),
                       torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # Forward pass through LSTM layer
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        predictions = self.linear(lstm_out.contiguous().view(batch_size, -1))
        return predictions


def train_model_1(df, epochs=3, learning_rate=0.01, run_model=True, sequence_length=14, is_scaled=False):
    #TODO early stopping
    #TODO run validation and test iterations
    #TODO save pickled model/binarys
    #GIST given by-the-minute data about a stock, train on 59 minutes and predict the 60th minute price

    losses = []
    preds = []
    targets = []


    batch_size = 20
    data = Data(df, sequence_length)
    data_load = DataLoader(data, batch_size=batch_size)

    model = Model(num_layers=2, input_dim=len(df.columns) - 1, seq_length=sequence_length)
    model = model.to(model.device)
    model.train()
    # x, y = data[0]
    # print(x.shape)
    # print(x)
    # print(y)
    # print(y.shape)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if is_scaled:
        scale_type = 'scaled'
    else:
        scale_type = 'not scaled'

    if run_model:
        for epoch in range(epochs):
            epoch_preds = []
            epoch_targets = []
            model.zero_grad()

            for idx, (x, y) in enumerate(data_load):
                x, y = x.to(device), y.to(device)

                model.init_hidden(x.size(0))

                y_pred = model(x)

                epoch_preds.append(y_pred.detach().numpy())
                epoch_targets.append(y.detach().numpy())

                y_pred = y_pred.to(model.device)
                loss = loss_function(y_pred, y)

                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()

            print('Epoch: {}.............'.format(epoch), end=' ')
            print("Loss: {:.4f}".format(loss.item()))

            preds.append(epoch_preds)
            targets.append(epoch_targets)

        plt.figure()
        plt.plot(losses, label='train loss')
        title = str('LSTM Loss Graph\n' + scale_type)
        plt.title(title)
        plt.legend()
        plt.savefig('figures/lstm_approach_1_loss_actual.png')
        plt.show()

        # plt.figure()
        # plt.plot(epoch_targets, label='targets')
        # plt.plot(epoch_preds, label='predictions')
        # title = str('LSTM Predictions Graph\n' + scale_type)
        # plt.title(title)
        # plt.legend()
        # # plt.savefig('figures/lstm_approach_1_predictions_actual.png')
        # plt.show()

        targets0 = np.concatenate(targets[0])
        preds0 = np.concatenate(preds[0])
        plt.figure()
        plt.plot(targets0, label='targets')
        plt.plot(preds0, label='predictions')
        title = str('LSTM Predictions Graph\n' + scale_type)
        plt.title(title)
        plt.legend()
        # plt.savefig('figures/lstm_approach_1_predictions_actual.png')
        plt.show()
    else:
        print('not running model')