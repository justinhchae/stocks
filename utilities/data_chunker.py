import numpy as np
import math

def index_marks(nrows, chunk_size):
    """
    a helper function for split()
    return an index of chunk size
    https://yaoyao.codes/pandas/2018/01/23/pandas-split-a-dataframe-into-chunks
    """
    return range(chunk_size, math.ceil(nrows / chunk_size) * chunk_size, chunk_size)

def split(dfm, chunk_size):
    """
    a helper function to split and chunk a dataframe by row
    :params: dfm -> a dataframe
    :params: chunk_size -> an inteter
    :returns: a list of chunked dataframes of size chunk_size
    """
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)

def chunk_data(train_data
                , price_col
                , time_col
                , n_prediction_units
                , window_size=15
                , seasonal_unit='day'
                , **kwargs
                ):
    """
    A helper function to chunk data into time windows
    seasonal_unit == day if working with sub daily dataset
    seasonal_unit == week if working with daily data within a weekly
    additional season units TODO
    """
    # these column names are required by the facebook api
    ds_col = 'ds'
    y_col = 'y'
    key_map = {time_col: ds_col, price_col: y_col}

    # full chunking intended for sub daily data such as minutes
    if seasonal_unit=='day':
        # extract the week number and day number for each timestamp for sorting
        train_data['week'] = train_data[time_col].dt.isocalendar().week
        train_data[seasonal_unit] = train_data[time_col].dt.isocalendar().day

        # produce a unique tuple (per year) of a week and day number
        train_data[seasonal_unit] = list(zip(train_data['week'], train_data[seasonal_unit]))
        train_data.drop(columns=['week'], inplace=True)

        # initialize valid test data
        valid = None
        # split data
        train = train_data

        # convert col names per facebook api needs
        train = train.rename(columns=key_map)

        # group df by seasonal unit as a tuple
        df = train.groupby(seasonal_unit)

        model_data = []

        for group_name, group_frame in df:
            chunk_data = []

            # in each seasonal_unit, chunk data into window_size chunks
            chunks = split(group_frame, window_size)

            # initialize an index to return each chunk in sequence
            idx = 0

            while 1:
                # return a data chunk of window_size on index idx
                chunk = chunks[idx]

                # set up index of next chunk in sequence
                next_idx = idx + 1

                # at then end of a seasonal_unit, break if index out of range
                if next_idx > len(chunks) - 1:
                    break
                else:
                    # otherwise, return the first n_prediction_units of next sequence as y target
                    target = chunks[next_idx].head(n_prediction_units)

                # increment the chunk
                idx += 1
                x_i = chunk.reset_index(drop=True)
                y_i = target.reset_index(drop=True)
                # save targets y and forecast predictions yhat
                chunk_data.append((x_i, y_i))

            model_data.append(chunk_data)

        return model_data

    # chunking daily data within the context of weeks
    if seasonal_unit == 'week':

        train_data = train_data.rename(columns=key_map)
        # chunk data into n sized sequences
        chunks = split(train_data, window_size)

        # initialize empty list
        chunked_data = []

        # enumerate through each chunk
        for idx, chunk in enumerate(chunks):

            # set up index of next chunk in sequence
            next_idx = idx + 1

            # at then end of a seasonal_unit, break if index out of range
            if next_idx > len(chunks) - 1:
                break
            else:
                # otherwise, return the first n_prediction_units of next sequence as y target
                target = chunks[next_idx].head(n_prediction_units)

            # set up tuples as train target pairs
            x_i = chunk.reset_index(drop=True)
            y_i = target.reset_index(drop=True)

            # save targets y and forecast predictions yhat
            chunked_data.append((x_i, y_i))

        return chunked_data

