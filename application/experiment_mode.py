from application.config import *
import os

def exp_mode():
    class_data_folder = os.sep.join([os.environ['PWD'], 'data', 'class_data'])

    if not os.path.exists(class_data_folder):
        pick_list_values = ['demo']
    else:
        # set available modes for data
        pick_list_values = ['demo', 'class_data']
    # convenience object
    sidebar_picklist = pick_list_values
    # return the select box with options as st object
    return st.sidebar.selectbox(label='Select Experiment Mode'
                                , options=sidebar_picklist
                                , key='exp_mode')