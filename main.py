from utilities.get_data import get_news_dummies, get_stock_dummies
from utilities.clean_data import trading_days
from utilities.run_arima import run_arima
from utilities.run_prophet import run_prophet
from utilities.data_chunker import chunk_data
from utilities.evaluate_model import assess_model
from utilities.prep_stock_data import split_stock_data, scale_stock_data
from model.lstm_approach_1 import train_model_1, train_model, test_model
from model.LSTM import Model
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp

if __name__ == '__main__':
    # run get pipelines for news and stock data
    news_df = get_news_dummies()
    stock_df = get_stock_dummies()

    # transform to minute-by-minute sentiment score and stock price
    df = trading_days(news_df, stock_df)

    # split data into train, validation, and testing
    train, valid, test = split_stock_data(df=df[['t','c']], time_col='t')
    # train lstm on unscaled data
    # train_model_1(train, run_model=False, is_scaled=False)

    # scale stock data
    train_scaled, valid_scaled, test_scaled, scaler = scale_stock_data(train=train
                                                                       , valid=valid
                                                                       , test=test
                                                                       )
    # START HERE uncomment the line you want to run; hide the rest
    # run_mode = 'arima'
    # run_mode = 'prophet'
    run_mode = 'lstm1'
    # run_mode = 'lstm2'

    # get count of max available CPUs, minus 2
    CPUs = cpu_count() - 2

    params = { 'train_data': train_scaled
              , 'valid_data': valid_scaled
              , 'test_data' : test_scaled
              , 'time_col': 't'
              , 'price_col': 'c'
              , 'run_model': True
              , 'window_size': 15
              , 'seasonal_unit':'day'
              , 'prediction_unit':'1min'
              , 'n_prediction_units':1}

    enable_mp = True

    if run_mode == 'arima':
        print(f'Training Approach run_mode: {run_mode}')
        # train arima on stock data only
        chunked_data = chunk_data(**params)
        print(f'Pooling {CPUs}x CPUs with Multiprocessor')
        # pooling enabled
        p = Pool(CPUs)
        results = list(tqdm(p.imap(run_arima, chunked_data)))
        p.close()
        p.join()

        # assess model results
        assess_model(results, model_type=run_mode, stock_name='Amazon')

    elif run_mode == 'prophet':
        print(f'Training Approach run_mode: {run_mode}')
        # train prophet on stock data only
        chunked_data = chunk_data(**params)
        print(f'Pooling {CPUs}x CPUs with Multiprocessor')
        # pooling enabled
        p = Pool(CPUs)
        results = list(tqdm(p.imap(run_prophet, chunked_data)))
        p.close()
        p.join()

        # assess model results
        assess_model(results, model_type=run_mode, stock_name='Amazon')

    elif run_mode == 'lstm1':
        print(f'Training Approach run_mode: {run_mode}')

        # train lstm on stock data only
        model = Model(num_layers=1, input_dim=1, seq_length=14)

        # preds = train_model_1(train_scaled, valid_scaled, test_scaled, model, epochs=20, run_model=True, is_scaled=True, sequence_length=14)

        if enable_mp:

            cores = 7
            mp.set_start_method('spawn')
            model.share_memory()

            processes = []
            print(f'Pooling {cores}x Cores with Multiprocessor')

            for rank in tqdm(range(cores)):
                # pool data for train_scaled to function train_model
                #TODO pin memory for GPU instance
                p = mp.Process(target=train_model, args=(train_scaled, model))
                p.start()
                processes.append(p)

            for p in tqdm(processes):
                p.join()
        else:
            print('Training Without Pooling')
            train_model(train=train_scaled, model=model)

        test_model(model, test_scaled)


    elif run_mode == 'lstm2':
        print('Training Approach run_mode:', run_mode)
        # split on data having closing price 'c' and sentiment score 'compound'
        model = Model(num_layers=1, input_dim=2, seq_length=14)

        # split data having both sentiment and stock price
        train, valid, test = split_stock_data(df=df, time_col='t')
        # train_model_1(train, valid, run_model=True, is_scaled=False)

        # scale data
        train_scaled, valid_scaled, test_scaled, scaler = scale_stock_data(train=train
                                                                           , valid=valid
                                                                           , test=test
                                                                           )
        # run model
        preds = train_model_1(train_scaled, valid_scaled, test_scaled, model, epochs=20, run_model=True, is_scaled=True, sequence_length=14)
