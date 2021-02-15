# adapted from
# https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca

import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_sequence(data, data_col, seq_len=59):
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
    # https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
    def __init__(self, input_size=1, hidden_layer_size=59, output_size=1, device="cpu"):
        super().__init__()

        if device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device =  torch.device("cpu")

        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size, device=self.device),
                            torch.zeros(1,1,self.hidden_layer_size, device=self.device))


    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        # predictions = self.linear(lstm_out.view(len(input_seq), -1))
        # return predictions[-1]
        return lstm_out, self.hidden_cell


def train_model_1(df, epochs=150):
    data = prep_arr(df, time_col='t', data_col='c')

    x_train, y_train = make_sequence(data=data, data_col='c')


    model = Model()
    model= model.to(model.device)
    model.train()

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_train = x_train.to(model.device)
    y_train = y_train.to(model.device)

    for i in range(epochs):
        losses = []
        optimizer.zero_grad()

        for idx, input in enumerate(x_train):

            # model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=model.device),
            #                      torch.zeros(1, 1, model.hidden_layer_size, device=model.device))

            # print('input', input[-3:])
            # print('target', y_train[idx][:3])
            output, state = model(input)
            model.hidden_cell = state
            predictions = model.linear(output.view(len(input), -1))
            y_pred = predictions[-1]
            # print('pred:', y_pred)

            y = torch.tensor([y_train[idx][0]])

            loss = loss_function(y_pred, y)
            # print('loss', loss)

            losses.append(loss.item())

            loss.backward(retain_graph=True)

            if idx == 100:
                break
        optimizer.step()

        plt.figure()
        plt.plot(losses)
        plt.show()
        break

        # complete a single sequence

        # if i%25 ==1:
        #     print(f'epoch: {i:3} loss: {loss.item():10.8f}')
        #
        # print(loss)




# def predict_model_1():
#     model.eval()
#
#     for i in range(fut_pred):
#         seq = torch.FloatTensor(test_inputs[-train_window:])
#         with torch.no_grad():
#             model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
#                             torch.zeros(1, 1, model.hidden_layer_size))
#             test_inputs.append(model(seq).item())





