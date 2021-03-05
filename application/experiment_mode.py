from application.config import *

def exp_mode():
    # set available modes for data
    pick_list_values = ['demo', 'class_data']
    # convenience object
    sidebar_picklist = pick_list_values
    # return the select box with options as st object
    return st.sidebar.selectbox(label='Select Experiment Mode'
                                , options=sidebar_picklist
                                , key='index_main_menu')