# adapted from
# https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_percentage_error

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def train_model_1(train, valid, test, model, epochs=10, learning_rate=0.001, run_model=True, sequence_length=14, is_scaled=False):
    #TODO early stopping
    #TODO run validation and test iterations
    #TODO save pickled model/binarys
    #GIST given by-the-minute data about a stock, train on 59 minutes and predict the 60th minute price

    losses = []
    losses_valid = []
    preds = []
    targets = []


    batch_size = 16
    train_set = Data(train, sequence_length)
    train_load = DataLoader(train_set, batch_size=batch_size)
    valid_set = Data(valid, sequence_length)
    valid_load = DataLoader(valid_set, batch_size=batch_size)
    test_set = Data(test, sequence_length)
    test_load = DataLoader(test_set, batch_size=batch_size)


    model = model.to(model.device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if is_scaled:
        scale_type = 'scaled'
    else:
        scale_type = 'not scaled'

    if run_model:
        for epoch in range(epochs):
            model.train(True)
            epoch_preds = []
            epoch_targets = []
            epoch_preds_valid = []
            epoch_targets_valid = []
            model.zero_grad()

            for idx, (x, y) in enumerate(train_load):
                x, y = x.to(model.device), y.to(model.device)

                model.init_hidden(x.size(0))

                y_pred = model(x)

                epoch_preds.append(y_pred.detach().numpy())
                epoch_targets.append(y.detach().numpy())

                y_pred = y_pred.to(model.device)
                loss = loss_function(y_pred, y)

                # losses.append(loss.item())
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()

            print('Epoch: {}.............'.format(epoch), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            losses.append(loss.item())

            with torch.no_grad():
                model.train(False)
                for idx, (x, y) in enumerate(valid_load):
                    x, y = x.to(model.device), y.to(model.device)

                    model.init_hidden(x.size(0))

                    y_pred = model(x)

                    epoch_preds_valid.append(y_pred.detach().numpy())
                    epoch_targets_valid.append(y.detach().numpy())

                    y_pred = y_pred.to(model.device)
                    loss_v = loss_function(y_pred, y)

                    # losses_valid.append(loss_v.item())

                    # optimizer.zero_grad()
                losses_valid.append(loss_v.item())



            preds.append(epoch_preds)
            targets.append(epoch_targets)

        test_preds = []
        test_targets = []
        with torch.no_grad():
            model.train(False)
            for idx, (x, y) in enumerate(test_load):
                x, y = x.to(model.device), y.to(model.device)

                model.init_hidden(x.size(0))

                y_pred = model(x)

                test_preds.append(y_pred.detach().numpy())
                test_targets.append(y.detach().numpy())
            test_preds = np.concatenate(test_preds)
            test_targets = np.concatenate(test_targets)
            mape = np.mean(np.abs((test_targets - test_preds) / test_targets)) * 100
            optimizer.zero_grad()

        print(f'MAPE score: {mape}')


        plt.figure()
        plt.plot(losses, label='train loss')
        plt.plot(losses_valid, label='valid loss')
        title = str('LSTM Loss Graph\n' + scale_type)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.savefig('figures/lstm_approach_1_loss_scaled.png')
        plt.show()


        plt.figure()
        # plt.plot(test.iloc[14:, 0], test_targets, label='targets')
        xrange = range(len(test_preds))
        plt.plot(test.iloc[14:, 0], test.iloc[14:, -1], label='targets', marker='x')
        plt.plot(test.iloc[14:, 0], test_preds, label='predictions', marker='x')
        title = str('LSTM Test Graph\n' + scale_type)
        plt.xlabel('Timestep')
        plt.ylabel('Scaled Price')
        plt.title(title)
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.savefig('figures/lstm_approach_1_test_scaled.png')
        plt.show()

        # targets0 = np.concatenate(targets[-1])
        # preds0 = np.concatenate(preds[-1])
        # plt.figure()
        # plt.plot(targets0, label='targets')
        # plt.plot(preds0, label='predictions')
        # title = str('LSTM Final Epoch Graph\n' + scale_type)
        # plt.title(title)
        # plt.legend()
        # plt.savefig('figures/lstm_approach_1_training_scaled.png')
        # plt.show()
    else:
        print('not running model')