from src.application.config import *
from src.utilities.get_data import get_stock_tickers

def tickers_mode(exp_mode):

    pick_list_values = []

    if exp_mode == 'demo':
        # set available modes for data
        pick_list_values = ['Amazon']

    elif exp_mode == 'class_data':

        pick_list_values = ['Run All']

        tickers_historical = None

        # try:
            # to run a single ticker, slice the get stock tickers function which returns a list
        historical_news_filename = 'news.json'
        class_data_folder = os.sep.join([os.environ['PWD'], 'data', 'class_data'])
        historical_news_path = os.sep.join([class_data_folder, historical_news_filename])
        tickers_historical = get_stock_tickers(historical_news_path)
            # debugging
        # except:
        #     pass

        if tickers_historical:
            pick_list_values.extend(tickers_historical)

    else:
        pick_list_values = ['Amazon']

    # convenience object
    sidebar_picklist = pick_list_values
    # return the select box with options as st object
    return st.sidebar.selectbox(label='Set Tickers', options=sidebar_picklist, key='tickers_mode')