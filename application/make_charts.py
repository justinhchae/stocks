from application.config import *
import altair as alt

import pandas as pd

def make_scatter(df, title):

    df = df.replace({' ':None})
    df['MAPE'] = df['MAPE'].astype('float32')

    df = df[df['MAPE']>0]

    st.write(title)

    c = alt.Chart(df).mark_circle().encode(alt.X('sentiment_variance'),
                                           alt.Y('price_variance'),
                                           size = 'MAPE'
                                         , color = 'model_type'
                                         , tooltip = ['ticker','model_type', 'MAPE', 'notes']
                                           ).transform_bin('MAPE','MAPE', bin=alt.Bin(maxbins=10))

    st.altair_chart(c, use_container_width=True)

def make_line():

    df = pd.read_csv('application/app_data/ABBV_arima_loss.csv')
    df = df.set_index('ds')

    st.line_chart(data=df, width=0, height=0, use_container_width=True)
