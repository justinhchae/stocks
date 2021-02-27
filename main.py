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
import torch
from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp

if __name__ == '__main__':

    stock = 'Amazon'
    # run get pipelines for news and stock data
    news_df = get_news_dummies(stock)
    stock_df = get_stock_dummies(stock)

    # transform to minute-by-minute sentiment score and stock price
    df = trading_days(news_df, stock_df)

    # split data into train, validation, and testing
    train, valid, test = split_stock_data(df=df[['t','c']], time_col='t')

    # scale stock data
    train_scaled, valid_scaled, test_scaled, scaler = scale_stock_data(train=train
                                                                       , valid=valid
                                                                       , test=test
                                                                       )
    # START HERE uncomment the line you want to run; hide the rest
    # run_mode = 'arima'
    run_mode = 'prophet'
    # run_mode = 'lstm1'
    # run_mode = 'lstm2'

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    enable_mp = True

    # configure parameters for forecasting here
    params = { 'stock_name': stock
              , 'train_data': train_scaled
              , 'valid_data': valid_scaled
              , 'test_data': test_scaled
              , 'time_col': 't'
              , 'price_col': 'c'
              , 'run_model': True
              , 'window_size': 15
              , 'seasonal_unit': 'day'
              , 'prediction_unit': '1min'
              , 'epochs': 5
              , 'n_layers': 1
              , 'learning_rate': 0.001
              , 'batch_size': 16
              , 'n_prediction_units': 1
              , 'device': device
              , 'max_cpu': cpu_count() // 2
              , 'pin_memory': False
               }

    #TODO: Conduct performance testing on optimal CPU count, currently pooling half of reported cpu_count

    if run_mode == 'arima':
        print('Forecasting for {} with Approach run_mode: {}'.format(params['stock_name'], run_mode))
        # train arima on stock data only
        chunked_data = chunk_data(**params)

        if enable_mp:
            print('Pooling {}x CPUs with Multiprocessor'.format(params['max_cpu']))
            # pooling enabled
            p = Pool(params['max_cpu'])
            results = list(tqdm(p.imap(run_arima, chunked_data)))
            p.close()
            p.join()
        else:
            print('Forecasting Without Pooling')
            results = [run_arima(i) for i in tqdm(chunked_data)]

        # assess model results with MAPE and visualize predict V target
        assess_model(results, model_type=run_mode, stock_name=params['stock_name'])

    elif run_mode == 'prophet':
        print('Forecasting for {} with Approach run_mode: {}'.format(params['stock_name'], run_mode))
        # train prophet on stock data only
        chunked_data = chunk_data(**params)

        if enable_mp:
            print('Pooling {}x CPUs with Multiprocessor'.format(params['max_cpu']))
            # pooling enabled
            p = Pool(params['max_cpu'])
            results = list(tqdm(p.imap(run_prophet, chunked_data)))
            p.close()
            p.join()
        else:
            print('Forecasting Without Pooling')
            results = [run_prophet(i) for i in tqdm(chunked_data)]

        # assess model results with MAPE and visualize predict V target
        assess_model(results, model_type=run_mode, stock_name=params['stock_name'])

    elif run_mode == 'lstm1':
        print('Forecasting for {} with Approach run_mode: {}'.format(params['stock_name'], run_mode))

        if is_cuda:
            params.update({'num_workers': 1
                           ,'pin_memory':True})

        params.update({'n_features':1})

        # train lstm on stock data only
        model = Model(num_layers=params['n_layers']
                      , input_dim=params['n_features']
                      , seq_length=params['window_size']
                      , device=params['device'])

        # preds = train_model_1(train_scaled, valid_scaled, test_scaled, model, epochs=20, run_model=True, is_scaled=True, sequence_length=14)

        if enable_mp:

            mp.set_start_method('spawn')
            model.share_memory()

            processes = []
            print('Pooling {}x CPUs with Multiprocessor'.format(params['max_cpu']))

            for rank in tqdm(range(params['max_cpu'])):
                # pool data for train_scaled to function train_model
                p = mp.Process(target=train_model
                               , args=(train_scaled
                                       , model
                                       , params['window_size']
                                       , params['pin_memory']
                                       , params['epochs']
                                       , params['learning_rate'])
                               )
                p.start()
                processes.append(p)

            for p in tqdm(processes):
                p.join()
        else:
            print('Forecasting Without Pooling')

            train_model(train=train_scaled
                        , model=model
                        , epochs=params['epochs']
                        , sequence_length=params['window_size']
                        , batch_size=params['batch_size']
                        , pin_memory=params['pin_memory']
                        )

        # assess model results with MAPE and visualize predict V target
        test_model(model=model
                   , dataset=test_scaled
                   , sequence_length=params['window_size']
                   , batch_size=params['batch_size']
                   , stock_name=params['stock_name']
                   , pin_memory=params['pin_memory']
                   )


    elif run_mode == 'lstm2':
        #TODO make a single lstm option with configurable data and params (combine lstm1 and lstm2)
        print('Forecasting for {} with Approach run_mode: {}'.format(params['stock_name'], run_mode))

        if is_cuda:
            params.update({'num_workers': 1
                           ,'pin_memory':True})

        params.update({'n_features': 2})

        model = Model(num_layers=params['n_layers']
                      , input_dim=params['n_features']
                      , seq_length=params['window_size']
                      , device=params['device']
                      )

        # split data having both sentiment and stock price
        train, valid, test = split_stock_data(df=df, time_col='t')

        # scale data
        train_scaled, valid_scaled, test_scaled, scaler = scale_stock_data(train=train
                                                                           , valid=valid
                                                                           , test=test
                                                                           )
        if enable_mp:

            mp.set_start_method('spawn')
            model.share_memory()

            processes = []
            print('Pooling {}x CPUs with Multiprocessor'.format(params['max_cpu']))

            for rank in tqdm(range(params['max_cpu'])):
                # pool data for train_scaled to function train_model
                #TODO pin memory for GPU instance
                p = mp.Process(target=train_model
                               , args=(train_scaled
                                       , model
                                       , params['window_size']
                                       , params['pin_memory']
                                       , params['epochs']
                                       , params['learning_rate'])
                               )
                p.start()
                processes.append(p)

            for p in tqdm(processes):
                p.join()
        else:

            print('Forecasting Without Pooling')

            train_model(train=train_scaled
                        , model=model
                        , epochs=params['epochs']
                        , sequence_length=params['window_size']
                        , batch_size=params['batch_size']
                        , pin_memory=params['pin_memory']
                        )

        test_model(model=model
                   , dataset=test_scaled
                   , sequence_length=params['window_size']
                   , batch_size=params['batch_size']
                   , stock_name=params['stock_name']
                   , pin_memory=params['pin_memory']
                   )
