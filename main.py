from utilities.get_data import get_news_dummies, get_stock_dummies, get_news_real, get_stock_real, get_stock_tickers
from utilities.clean_data import trading_days
from utilities.run_arima import run_arima
from utilities.run_prophet import run_prophet
from utilities.data_chunker import chunk_data
from utilities.evaluate_model import assess_model
from utilities.prep_stock_data import split_stock_data, scale_stock_data
from model.lstm_approach_1 import train_model, test_model, plot_losses
from model.LSTM import Model
from tqdm import tqdm
import torch
# from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
from functools import partial

if __name__ == '__main__':
    # initialize empty list to hold stock refs
    tickers = []

    # configure gpu if available
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    mp.set_start_method('spawn')

    ## make two options to run the program, one for experiment mode and one run modes

    # uncomment one of two types of exp modes
    # experiment_mode = 'class_data'
    experiment_mode = 'demo'

    # uncomment one of two types of run modes
    run_modes = ['arima', 'prophet']
    # run_modes = ['lstm1', 'lstm2']

    if experiment_mode == 'demo':
        tickers = ['Amazon']
        news_df = get_news_dummies(tickers[0])
        stock_df = get_stock_dummies(tickers[0])
        # transform to minute-by-minute sentiment score and stock price
        df = trading_days(news_df, stock_df)

    elif experiment_mode == 'class_data':
        tickers = get_stock_tickers()
        # later, cycle through tickers, for now, work with first ticker in index
        news_df = get_news_real(ticker=tickers[0])
        stock_df = get_stock_real(ticker=tickers[0])
        #TODO: design how to combine data for experiement
        print(stock_df)
        breakpoint()

    # split data into train, validation, and testing
    train, valid, test = split_stock_data(df=df[['t','c']], time_col='t')

    # scale stock data
    train_scaled, valid_scaled, test_scaled, scaler = scale_stock_data(train=train
                                                                       , valid=valid
                                                                       , test=test
                                                                       )

    for run_mode in run_modes:

        # configure parameters for forecasting here
        params = { 'stock_name': tickers[0]
                  , 'train_data': train_scaled
                  , 'valid_data': valid_scaled
                  , 'test_data': test_scaled
                  , 'time_col': 't'
                  , 'price_col': 'c'
                  , 'run_model': True
                  , 'window_size': 15
                  , 'seasonal_unit': 'day'
                  , 'prediction_unit': '1min'
                  , 'epochs': 6
                  , 'n_layers': 1
                  , 'learning_rate': 0.001
                  , 'batch_size': 16
                  , 'n_prediction_units': 15
                  , 'device': device
                  , 'max_processes': mp.cpu_count() // 2
                  , 'pin_memory': False
                  , 'enable_mp': True
                  , 'run_mode':run_mode
                   }

        if params['run_mode'] == 'arima' or params['run_mode'] == 'prophet':
            print('Forecasting for {} with Approach run_mode: {}'.format(params['stock_name'], run_mode))
            # train arima on stock data only
            chunked_data = chunk_data(**params)

            # the model_ object is a temporary object to be updated during mp with partial()
            model = run_arima if run_mode == 'arima' else run_prophet if run_mode == 'prophet' else None

            if params['enable_mp']:
                print('Pooling {}x Processes with Multiprocessor'.format(params['max_processes']))
                # pooling enabled
                p = mp.Pool(params['max_processes'])
                # pass function params with partial method, then call the partial during mp
                model_ = partial(model, n_prediction_units=params['n_prediction_units'])
                results = list(tqdm(p.imap(model_, chunked_data)))
                p.close()
                p.join()
            else:
                print('Forecasting Without Pooling')
                results = [model(i, n_prediction_units=params['n_prediction_units']) for i in tqdm(chunked_data)]

            # assess model results with MAPE and visualize predict V target
            assess_model(results, model_type=run_mode, stock_name=params['stock_name'])

        elif params['run_mode'] == 'lstm1' or params['run_mode'] =='lstm2':

            print('Forecasting for {} with Approach run_mode: {}'.format(params['stock_name'], run_mode))

            if is_cuda:
                params.update({'num_workers': 1
                               ,'pin_memory':True})

            n_features = 1 if run_mode == 'lstm1' else 2 if run_mode == 'lstm2' else None

            # if run mode == lstm2, then refactor data to have both sentiment and price
            if run_mode == 'lstm2':
                # split data having both sentiment and stock price
                train, valid, test = split_stock_data(df=df, time_col='t')
                # scale data
                train_scaled, valid_scaled, test_scaled, scaler = scale_stock_data(train=train
                                                                                   , valid=valid
                                                                                   , test=test)

                params.update({'train_data': train_scaled
                              , 'valid_data': valid_scaled
                              , 'test_data': test_scaled
                              , 'scaler': scaler
                               })

            params.update({'n_features':n_features})

            params.update({'input_dim':params['n_features']
                          , 'seq_length':params['window_size']
                          , 'sequence_length':params['window_size']
                           })

            model = Model(**params)

            params.update({'model': model})

            # force no multiprocessing due to performance issues
            params.update({'enable_mp': False})

            if params['enable_mp']:
                params['model'].share_memory()

                processes = []
                print('Pooling {}x Processes with Multiprocessor'.format(params['max_processes']))

                for rank in tqdm(range(params['max_processes'])):
                    # pool data for train_scaled to function train_model
                    p = mp.Process(target=train_model, kwargs=params)
                    p.start()
                    processes.append(p)

                for p in tqdm(processes):
                    p.join()
            else:
                print('Forecasting Without Pooling')
                # run train, validation, and test
                train_model(**params)

