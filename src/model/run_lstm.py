# adapted from
# https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_percentage_error

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

class Data(Dataset):
    def __init__(self, data, window, yhat='c', sentiment='compound', step_size=1):
        self.data = data
        self.window = window
        self.step_size = step_size
        self.sentiment = sentiment
        # self._xdata = # index of times
        self.yhat = data.columns.get_loc(yhat)
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        x = self.data.iloc[index: index + self.window, 1:]
        y = self.data.iloc[index + self.window, self.yhat:self.yhat+1]

        # set sentiment score to that of the one corresponding to the target price
        if self.sentiment in self.data.columns:
            y_sentiment = self.data[self.sentiment].iloc[index + self.window]
            x[self.sentiment] = y_sentiment

        return torch.tensor(x.to_numpy(), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.window

    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def __getsize__(self):
        return (self.__len__())

def train_model(train_data, model, sequence_length, pin_memory, epochs=20, learning_rate=0.001, batch_size=16, **kwargs):
    # new train function, replace train_model1 with this
    # https://pytorch.org/docs/stable/notes/multiprocessing.html
    # TODO: don't do this, fix how to better pass results folder
    # if kwargs['model_results_folder']:
    #     model_results_folder = kwargs['model_results_folder']
    # else:
    model_results_folder = None

    # return data_loader objects for train, validation, and test
    train_set = Data(train_data, sequence_length)
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=pin_memory)

    valid_set = Data(kwargs['valid_data'], sequence_length)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    test_set = Data(kwargs['test_data'], sequence_length)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # set loss function
    loss_function = nn.MSELoss()
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # init empty lists for overall loss values
    train_losses = []
    valid_losses = []

    # init a progress bar object of range epoch
    # ref: https://stackoverflow.com/questions/37506645/can-i-add-message-to-the-tqdm-progressbar
    pbar = tqdm(range(epochs), desc='Epoch', position=0, leave=True)

    # patience value for early stopping
    patience = 2
    # get a true count of actual epochs run
    actual_run_epochs = 0
    # absolute max epochs to run
    max_epochs = epochs * 2
    # threshold for min variance
    min_variance = 0.000001
    # stopping reason for tracking
    stop_reason = 'ran all planned epochs'
    # iterate through n pbars
    for epoch in pbar:
        # if the epoch is reset more than max_epochs, break
        if actual_run_epochs > max_epochs:
            stop_reason = 'model could not converge'
            break
        # increment a counter to track actual run epochs
        actual_run_epochs += 1
        # run train epoch, return losses
        train_loss = train_epoch(model, train_loader, loss_function, optimizer)
        train_losses.append(train_loss)

        # run validation epoch, return losses
        valid_loss = valid_epoch(model, valid_loader, loss_function)
        valid_losses.append(valid_loss)

        # compute ave epoch losses for print out
        epoch_train_loss = np.mean(train_loss)
        epoch_valid_loss = np.mean(valid_loss)

        # update and refresh progress bar each epoch
        pbar.set_description('{}-{} Epoch {}...Mean Train Loss: {:.5f}...Mean Valid Loss: {:.5f}'.format(kwargs['stock_name'],kwargs['run_mode'], epoch, epoch_train_loss, epoch_valid_loss))
        pbar.refresh()

        # testing some early stopping criteria
        if epoch > patience:
            curr_loss = valid_losses[-1]
            last_loss = valid_losses[-2]
            try:
                last_prior_loss = valid_losses[-3]
            except:
                last_prior_loss = 1000000

            # ave_loss = np.mean(valid_losses)
            last_n_losses = valid_losses[-2:]
            variance = np.var(last_n_losses)

            # if loss increases, break if the ave loss is below target loss threshold
            if curr_loss > last_loss and curr_loss > last_prior_loss:
                # stop training if the epoch loss increases
                stop_reason = 'loss started increasing'
                break

            elif variance < min_variance:
                # stop training if loss effectively stops changing
                stop_reason = 'loss stopped changing'
                break

    # plot losses
    if model_results_folder is not None:

        plot_losses(train_loss=train_losses
                    , valid_loss=valid_losses
                    , stock_name=kwargs['stock_name']
                    , model_type=kwargs['run_mode']
                    , model_results_folder=model_results_folder
                    )

    # test model
    results = test_model(model
               , test_loader
               , stock_name=kwargs['stock_name']
               , model_type=kwargs['run_mode']
               , loss_function=loss_function
               , test_data=kwargs['test_data']
               , n_hidden=kwargs['hidden_dim']
               , n_epochs=actual_run_epochs
               , stop_reason=stop_reason
               , model_results_folder=model_results_folder
               )

    return results

def valid_epoch(model, data_loader, loss_function):
    # initiate empty list to hold loss values
    losses = []

    # enumerate through nn without grads and no training
    with torch.no_grad():
        model.train(False)
        for idx, (x, y) in enumerate(data_loader):
            # get objects on device
            x, y = x.to(model.device), y.to(model.device)
            # init hidden states
            model.init_hidden(x.size(0))
            # return model prediction
            y_pred = model(x)
            # objects to device for loss
            y_pred = y_pred.to(model.device)
            loss = loss_function(y_pred, y)
        # capture losses and return as a list
        losses.append(loss.item())

    return losses

def train_epoch(model, data_loader, loss_function, optimizer):
    # set model to train and zero gradients
    model.train(True)
    model.zero_grad()

    # store losses for each iter of the epoch
    losses = []

    # unpack index and x, y values for each data object
    for idx, (x, y) in enumerate(data_loader):
        # get objects on device
        x, y = x.to(model.device), y.to(model.device)

        # init hidden layers
        model.init_hidden(x.size(0))

        # get prediction
        y_pred = model(x)

        # evaluate loss
        y_pred = y_pred.to(model.device)
        loss = loss_function(y_pred, y)
        loss.backward()

        # step up
        optimizer.step()
        optimizer.zero_grad()

        # capture and return losses
        losses.append(loss.item())

    return losses

def plot_losses(train_loss, valid_loss, stock_name, model_type, model_results_folder):

    # return averages for each epoch loss
    ave_train_loss = [np.mean(i) for i in train_loss]
    ave_valid_loss = [np.mean(i) for i in valid_loss]

    # set figure and plot data
    fig, ax, = plt.subplots()
    ax.plot(ave_train_loss, color='orange', label='Train Loss')
    ax.plot(ave_valid_loss, color='blue', label='Valid Loss')

    ax.set_title(f'{stock_name} Stock Price Prediction Loss\nWith {model_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    out_filename = f'{stock_name}_{model_type}_loss.png'
    out_path = os.sep.join([model_results_folder, out_filename])
    fig.savefig(out_path)
    # plt.show()

    df = pd.DataFrame()
    df['train_loss'] = ave_train_loss
    df['validation_loss'] = ave_valid_loss

    df['model_type'] = model_type
    out_filename = f'{stock_name}_{model_type}_loss.csv'
    out_path = os.sep.join([model_results_folder, out_filename])
    df.to_csv(out_path, index=False)

def test_model(model, data_loader, stock_name, model_type, loss_function, test_data, n_hidden, stop_reason, n_epochs, model_results_folder):

    # initialize empty lists to capture data
    losses = []
    targets = []
    predictions = []

    # enumerate through nn without grads and no training
    with torch.no_grad():
        # set model to NOT training
        model.train(False)

        for idx, (x, y) in enumerate(data_loader):
            # get objects on device
            x, y = x.to(model.device), y.to(model.device)
            # init hidden states
            model.init_hidden(x.size(0))
            # return model prediction
            y_pred = model(x)
            # objects to device for loss
            y_pred = y_pred.to(model.device)
            # evaluate loss and store it
            loss = loss_function(y_pred, y)
            losses.append(loss.item())
            # detach values as numpy arrays
            predictions.append(y_pred.detach().numpy())
            targets.append(y.detach().numpy())

    # collapse each array into 1D arrays (list-like)
    predictions = np.concatenate(predictions).flatten()
    targets = np.concatenate(targets).flatten()

    # evaluate average MSE
    average_loss = np.mean(losses)
    # print('Mean Test Loss (MSE): {:.5f}'.format(average_loss))

    # evaluate MAPE
    eval_targets = test_data['c'].values
    eval_targets = eval_targets[:len(predictions)]
    error = mean_absolute_percentage_error(eval_targets, predictions) * 100
    # print('Error (MAPE): {:.5f}'.format(error))

    # organize data for plotting, pandas for convenience
    df = pd.DataFrame()
    df['targets'] = targets
    df['predictions'] = predictions

    if model_results_folder is not None:

        # plot the data
        fig, ax, = plt.subplots()

        # the actual data
        ax.plot(test_data['t']
              , test_data['c']
              , color='red'
              , label='Scaled Price'
                )

        # the predicted data at the equivalent index of target
        ax.plot(test_data['t'][:len(predictions)]
              , predictions
              , color='blue'
              , marker='o'
              , markersize=3
              , linestyle='dashed'
              , linewidth=1
              , label='Predicted Price'
               )

        h = n_hidden
        ax.set_title('{} Stock Price Prediction | Hidden: {} | Epochs: {}\nWith {}, Test MAPE: {:.4f}, Mean Test Loss:{:.4f}'.format(stock_name, h, n_epochs,model_type, error, average_loss))
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend(loc="upper left")
        fig.autofmt_xdate()
        plt.tight_layout()
        out_filename = f'{stock_name}_{model_type}_results.png'
        out_path = os.sep.join([model_results_folder, out_filename])
        fig.savefig(out_path)
        # plt.show()



        df['model_type'] = model_type
        out_filename = f'{stock_name}_{model_type}_results.csv'
        out_path = os.sep.join([model_results_folder, out_filename])
        df.to_csv(out_path, index=False)
        # df.to_csv(f'data/model_results/{stock_name}_{model_type}_results.csv', index=False)

    results = {'ticker': stock_name
        , 'N': len(test_data)
        , 'MAPE': error
        , 'date_start': min(test_data['t'])
        , 'date_end': max(test_data['t'])
        , 'model_type': model_type
        , 'notes': stop_reason
        , 'n_epochs': n_epochs
               }
    return results
