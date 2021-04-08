from src.runners.run_main import run_main
from src.utilities.get_data import get_stock_tickers

import os

if __name__ == '__main__':
    # uncomment one of two types of experiment modes
    # experiment_mode = 'class_data'
    experiment_mode = 'demo'
    demo_ticker = 'Amazon'
    data_folder = os.sep.join([os.environ['PWD'], 'data'])

    # if debug mode is True, script will run normally without multiprocessing
    # set to False to run without multiprocessing, this is helpful for debugging things in serial
    debug_mode = False

    # initialize empty structs
    tickers_historical = None

    historical_news_filename = 'news.json'
    class_data_folder = os.sep.join([data_folder, 'class_data'])
    historical_news_path = os.sep.join([class_data_folder, historical_news_filename])
    historical_stock_path = os.sep.join([class_data_folder, 'historical_price'])

    if os.path.exists(class_data_folder) & os.path.exists(historical_news_path):
        # only run experiment on class data (non-public data) if these paths exist
        tickers_historical = get_stock_tickers(historical_news_path)

    # set up tickers iterable object
    tickers = tickers_historical if tickers_historical is not None else [demo_ticker]

    # folder paths for data and models
    model_results_folder = os.sep.join([data_folder, 'model_results'])
    if not os.path.exists(model_results_folder):
        os.makedirs(model_results_folder)

    run_main(experiment_mode=experiment_mode
           , tickers=tickers
           , debug_mode=debug_mode
           , data_folder=data_folder
           , model_results_folder=model_results_folder
           , historical_news_path=historical_news_path
           , historical_stock_path=historical_stock_path)
