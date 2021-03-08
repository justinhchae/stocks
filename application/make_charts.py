from application.config import *
import altair as alt

import pandas as pd

def make_scatter(df, title):

    df = df.replace({' ':None})
    df['MAPE'] = df['MAPE'].astype('float32')

    df = df[df['MAPE']>0]

    st.write(title)

    c = alt.Chart(df).mark_point().encode(alt.X('sentiment_variance'),
                                           alt.Y('price_variance'),
                                           size = 'MAPE'
                                         , color = 'model_type'
                                         , tooltip = ['ticker','model_type', 'MAPE', 'notes']
                                           ).transform_bin('MAPE','MAPE', bin=alt.Bin(maxbins=10))
    c.configure_mark(
        opacity=.5

    )
    st.altair_chart(c, use_container_width=True)

def make_histogram(filename, group_by='model_type'):

    df = pd.read_csv(filename)
    df = df.replace({' ': None})
    df['MAPE'] = df['MAPE'].astype('float32')

    overall = df.groupby(group_by).agg('mean')
    st.write('Table of Data for Experiment')
    st.table(overall)
    grouped = df.groupby(group_by)

    x = None
    cols = st.beta_columns(len(grouped))

    for idx, (name, group) in enumerate(grouped):
        x = group.reset_index(drop=True).sort_values(by=['MAPE'])
        low = x.head(1)[['ticker', 'MAPE']]
        high = x.tail(1)[['ticker', 'MAPE']]
        c = alt.Chart(x).mark_bar().encode(alt.X("MAPE", bin=True), y='count()')
        cols[idx].write(f'MAPE Distribution for {name}')
        cols[idx].write('high')
        cols[idx].table(high)
        cols[idx].write('low')
        cols[idx].table(low)
        cols[idx].altair_chart(c, use_container_width=True)

    # for idx, (name, group) in enumerate(grouped):
    #     x = group.reset_index(drop=True).sort_values(by=['MAPE'])
    #     high = x.head(1)[['ticker', 'MAPE']]
    #     low = x.tail(1)[['ticker', 'MAPE']]
    #     st.write('high', high)
    #     st.write('low', low)

        # break



