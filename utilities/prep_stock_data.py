from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.autograd import Variable
import torch

def split_stock_data(df, time_col, pct_train=.7, pct_valid=.15, pct_test=.15):
    """
    split into train, validation, test
    :params df: a dataframe of stock data
    :params time_col: a string, the col representing time (must be pandas datetime)
    :returns: a split of stock data into train, validation, and test
    :
    """
    df = df.sort_values(by=time_col)
    N = len(df)

    # 70% of data for test, 15% for validation and 15% for test
    # initial model mvp: train on first part of data, predict rest
    # next phase: train/test on batches throughout dataset, i.e. 4-5 days at a time

    n_train = int(pct_train * N)
    n_valid = int(pct_valid * N)
    n_test = int(pct_test * N)

    subtotal = n_train + n_valid + n_test

    if subtotal != N:
        # account for any remaining rows, add them to test
        diff = N-subtotal
        n_test += diff
        subtotal = n_train + n_valid + n_test

    train = df.iloc[:n_train, :].copy()
    valid = df.iloc[n_train:n_train+n_valid, :].copy()
    test = df.iloc[n_train+n_valid:, :].copy()

    print('Split stock data int train, validation, and test.')

    return train, valid, test

def scale_stock_data(train, valid, test, cols=None):
    # returning the scaler allows for doing .inverse_transform()
    if cols is None:
        # scale all but the time column
        cols = ['v', 'vw', 'o', 'c', 'h', 'l', 'n']

    scaler = StandardScaler()

    train[cols] = scaler.fit_transform(train[cols])
    valid[cols] = scaler.transform(valid[cols])
    test[cols] = scaler.transform(test[cols])

    print('Scaled stock data. Fit_transform on train, transformed validation, and test.')


    return train, valid, test, scaler


def prep_arr(df, time_col, data_col):
    """
    this function deprecated and not currently used, keeping for comparison purposes
    usage data = prep_arr(df, time_col='t', data_col='c')
    :param df: a pandas df having time cols and data cols
    :param time_col: a string of the time col
    :param data_col: a string of the date col
    :return:
    """
    data_dict = {}

    time_index = df[time_col].values
    data_values = df[data_col].values

    data_dict.update({time_col:time_index})
    data_dict.update({data_col:data_values})

    return data_dict


def make_sequence(data, data_col, seq_len, step_size=1):
    """
    usage
    this function deprecated and not currently used, keeping for comparison purposes
    # x_train, y_train = make_sequence(data=data, data_col='c')

    # x_train = x_train.to(model.device)
    # y_train = y_train.to(model.device)

    return sequenced stock data as torch tensors
    of shape len(data) by sequence length
    """
    x, y = [], []

    arr = data[data_col]

    for i in range(len(arr) - seq_len):
        x_i = arr[i: i + seq_len]
        # TODO, fix slicing
        y_i = arr[i + step_size: i + seq_len + step_size]

        x.append(x_i)
        y.append(y_i)

    x_arr = np.array(x).reshape(-1, seq_len)
    y_arr = np.array(y).reshape(-1, seq_len)

    x_var = Variable(torch.from_numpy(x_arr).float())
    y_var = Variable(torch.from_numpy(y_arr).float())

    return x_var, y_var


