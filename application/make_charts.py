from application.config import *

def make_histograms(df):
    st.write('making stuff here')
    group_labels = list(df['model_type'].unique())
    st.write(group_labels)


