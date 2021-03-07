from application.config import *
import altair as alt
import pandas as pd

def make_scatter(df, title):

    df = df.replace({' ':None})
    df['MAPE'] = df['MAPE'].astype('float32')

    df = df[df['MAPE']>0]

    st.write(title)

    c = alt.Chart(df).mark_circle().encode(alt.X('sentiment_variance'),
                                           alt.Y('MAPE', bin=True),
                                           size = 'N'
                                         , color = 'model_type'
                                         , tooltip = ['ticker','model_type', 'MAPE', 'notes']
                                           )\
        #.transform_bin('sentiment_variance', 'sentiment_variance', bin=alt.Bin(maxbins=8))

    st.altair_chart(c, use_container_width=True)



