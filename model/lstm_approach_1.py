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
sequence_length = 59

try:
  print('Number of GPUs:',torch.cuda.device_count())
  print('GPU Card:', torch.cuda.get_device_name(0))
  device = torch.device('cuda:0')
except:
  print("No GPU this session, learning with CPU.")
  device = torch.device("cpu")


def make_sequence(data, data_col, seq_len=sequence_length, step_size=1):
    """
    return sequenced stock data as torch tensors
    of shape len(data) by sequence length
    """
    x, y = [], []

    arr = data[data_col]

    for i in range(len(arr) - seq_len):
        x_i = arr[i : i + seq_len]
        # TODO, fix slicing
        y_i = arr[i + step_size : i + seq_len + step_size]

        x.append(x_i)
        y.append(y_i)

    x_arr = np.array(x).reshape(-1, seq_len)
    y_arr = np.array(y).reshape(-1, seq_len)

    x_var = Variable(torch.from_numpy(x_arr).float())
    y_var = Variable(torch.from_numpy(y_arr).float())

    return x_var, y_var

#TODO, implement dataloader / dataset class for batching and training.

class Data(Dataset):
    def __init__(self, data, window, step_size=1):
        self.data = data
        self.window = window
        self.step_size = step_size
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        x = self.data.iloc[index: index + self.window, 1:]
        y = self.data.iloc[index + self.window + self.step_size, 1:]
        return torch.tensor(x.to_numpy(), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.window

    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def __getsize__(self):
        return (self.__len__())


def prep_arr(df, time_col, data_col):
    data_dict = {}

    time_index = df[time_col].values
    data_values = df[data_col].values

    data_dict.update({time_col:time_index})
    data_dict.update({data_col:data_values})

    batch_size = 20
    seq_length = sequence_length
    pin_memory = True
    num_workers = 4

    arr = np.array(data_values)

    dataset = Data(arr, seq_length)
    print(dataset[0])

    # data_load = DataLoader(dataset
    #                        , batch_size=batch_size
    #                        , drop_last=True
    #                        , num_workers=num_workers
    #                        , pin_memory=pin_memory)
    #
    # print(data_load)
    print(data_dict)
    return data_dict


class Model(nn.Module):

    def __init__(self
                 , input_dim=sequence_length
                 , hidden_dim=sequence_length
                 , output_dim=1
                 , num_layers=1
                 , device=device
                 ):

        super(Model, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, 1, self.hidden_dim, device=self.device),
                torch.zeros(self.num_layers, 1, self.hidden_dim, device=self.device))

    def forward(self, x):
        # Forward pass through LSTM layer

        lstm_out, self.hidden = self.lstm(x.view(len(x), 1, -1), self.hidden)
        # Only take the output from the final timestep

        predictions = self.linear(lstm_out.view(len(x), -1))
        return predictions


def train_model_1(df, epochs=3, learning_rate=0.01, run_model=True):

    #TODO: batching, nn layers, early stopping

    losses = []
    preds = []
    targets = []

    model = Model(num_layers=2)
    model = model.to(model.device)
    model.train()

    # data = prep_arr(df, time_col='t', data_col='c')
    data = Data(df, sequence_length)
    data_load = DataLoader(data
                           , batch_size=20)

    # x_train, y_train = make_sequence(data=data, data_col='c')

    # x_train = x_train.to(model.device)
    # y_train = y_train.to(model.device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if run_model:
        for epoch in range(2):
            epoch_preds = []
            epoch_targets = []
            for idx, (x, y) in enumerate(data_load):
                if idx == 1000:
                    break
                x, y = x.to(device), y.to(device)

                model.zero_grad()
                model.hidden = model.init_hidden()

                y_pred = model(x)
                #FIXME (when reshaping the y target slicing)

                epoch_preds.append(y_pred.detach().numpy())
                # y = torch.tensor([y_train[idx][-1]])
                epoch_targets.append(y.detach().numpy())


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
            preds.append(epoch_preds)
            targets.append(epoch_targets)

        plt.figure()
        plt.plot(losses, label='train loss')
        plt.title('loss graph lstm approach 1\nwith actual prices')
        plt.legend()
        plt.savefig('figures/lstm_approach_1_loss_actual.png')
        plt.show()

        targets0 = np.concatenate(targets[0])
        preds0 = np.concatenate(preds[0])
        plt.figure()
        plt.plot(targets0, label='targets')
        plt.plot(preds0, label='predictions')
        plt.title('price prediction to target lstm approach 1\nwith actual prices')
        plt.legend()
        # plt.savefig('figures/lstm_approach_1_predictions_actual.png')
        plt.show()

        targets0 = np.concatenate(targets[1])
        preds0 = np.concatenate(preds[1])
        plt.figure()
        plt.plot(targets0, label='targets')
        plt.plot(preds0, label='predictions')
        plt.title('price prediction to target lstm approach 1\nwith actual prices')
        plt.legend()
        # plt.savefig('figures/lstm_approach_1_predictions_actual.png')
        plt.show()
    else:
        print('not running model')
