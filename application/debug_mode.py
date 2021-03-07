from application.config import *

def debug_mode():
    # set available modes for data
    pick_list_values = [True, False]
    # convenience object
    sidebar_picklist = pick_list_values
    # return the select box with options as st object
    return st.sidebar.selectbox(label='Set Debug Mode (Default to True)'
                                , options=sidebar_picklist
                                , key='debug_mode')