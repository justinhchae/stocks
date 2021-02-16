from sklearn.preprocessing import StandardScaler
import numpy as np

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


