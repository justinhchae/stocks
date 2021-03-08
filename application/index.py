from application.config import *
from application.experiment_mode import exp_mode
from application.debug_mode import debug_mode
from application.default_run_modes import default_runs
from application.make_charts import make_scatter, make_histogram
from utilities.get_data import get_stock_tickers

import pandas as pd

from main import main

class Application():
    def __init__(self):
        # the text that displays in the tab at the very top of the page
        st.set_page_config(page_title='Stock Forecasting')
        self.sample_chart = 'application/app_data/results_app_scatter.csv'

    def run_app(self):
        # primary app call will run everything contained in frame()
        self.frame()

    def frame(self):
        # place main components of page here, add more as necessary
        self.title()
        self.body()
        self.footer()

    def title(self):
        # execute st calls for title section
        st.title('Experimenting With Stock Price Prediction')

    def body(self):
        # execute st calls for body section
        # a header for this section
        sub_title = 'Application Experiment for Advanced Deep Learning'
        st.markdown(f"<h3 style='text-align: center; color: black;font-family:courier;'>{sub_title}</h3>", unsafe_allow_html=True)
        # display some overview graphs
        my_expander = st.beta_expander("Experiment Results (Click to Hide or Show)", expanded=True)
        with my_expander:
            df = pd.read_csv(self.sample_chart)
            make_scatter(df=df, title='High-level Results for Error by Sentiment and Price Variance')
            make_histogram(filename=self.sample_chart)

        # makes a sidebar selection in index
        experiment_mode = exp_mode()

        debug_type = debug_mode()

        demo_run_mode = None

        if experiment_mode == 'demo':
            tickers = ['Amazon']
            run_modes_selection = default_runs()
            st.write('Default Run Modes set to:', run_modes_selection)
            if run_modes_selection == 'Baseline (ARIMA and FB Prophet)':
                demo_run_mode = ['arima', 'prophet']
            elif run_modes_selection == 'Featured (LSTMs)':
                demo_run_mode = ['lstm1', 'lstm2']
        else:
            tickers = get_stock_tickers()

        st.write('Experiment Configuration:')
        st.write('Experiment Mode:', experiment_mode)
        st.write('Debug Mode:',  debug_type)
        st.write('Tickers:', len(tickers))

        run_exp = st.button('Run!')

        results_df = None

        if run_exp:
            if not debug_type:
                st.write('Running with Multiprocessor')

            with st.spinner('Running The Experiment!...'):
                if demo_run_mode:
                    results_df = main(experiment_mode=experiment_mode, tickers=tickers, debug_mode=debug_type, demo_run_mode=demo_run_mode)

                else:
                    results_df = main(experiment_mode=experiment_mode, tickers=tickers, debug_mode=debug_type)

            st.success('Yay! Made Predictions.')

            st.write('Check data/ and figures/ for results')

        if results_df is not None:
            st.dataframe(results_df)
            make_scatter(df=results_df, title='Your Experiment Results')
            st.balloons()

    def footer(self):
        # make st calls for footer section here
        version_status = 'Version Alpha'
        st.markdown(
            f'<i style="font-size:11px">{version_status}</i>',
            unsafe_allow_html=True)
        owner_url = 'https://github.com/justinhchae/stocks'
        st.markdown(
            f'<i style="font-size:11px">&copy All Rights Reserved [The Project Group]({owner_url})</i>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:11px">The information provided by this app (the “Site”) is for general informational purposes only. All information on the Site is provided in good faith, however we make no representation or warranty of any kind, express or implied, regarding the accuracy, adequacy, validity, reliability, availability or completeness of any information on the Site.</p>',
            unsafe_allow_html=True
        )
