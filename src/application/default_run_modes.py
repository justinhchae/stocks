from application.config import *

def default_runs():
    # set available modes for data
    pick_list_values = ['Baseline (ARIMA and FB Prophet)', 'Featured (LSTMs)']
    # convenience object
    sidebar_picklist = pick_list_values
    # return the select box with options as st object
    return st.sidebar.selectbox(label='Select Default Run Mode'
                                , options=sidebar_picklist
                                , key='default_runs')