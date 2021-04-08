from src.application.config import *
from src.application.experiment_mode import exp_mode
from src.application.debug_mode import debug_mode
from src.application.default_run_modes import default_runs
from src.application.make_charts import make_scatter, make_histogram
from src.application.latex_equations import lstm_math, arima_math, prophet_math
from src.application.narratives import narrative_introduction, narrative_conclusion

from src.utilities.get_data import get_stock_tickers

import pandas as pd
from PIL import Image

from main import run_main

class Application():
    def __init__(self):
        # the text that displays in the tab at the very top of the page
        st.set_page_config(page_title='Stock Forecasting')
        app_data_folder = os.sep.join([os.environ['PWD'], 'src', 'application', 'app_data'])

        self.sample_chart = os.sep.join([app_data_folder, 'results_app_scatter.csv'])

        self.seq_1 = Image.open(os.sep.join([app_data_folder, 'sequence_1.png']))
        self.seq_2 = Image.open(os.sep.join([app_data_folder, 'sequence_2.png']))

        self.data_1 = Image.open(os.sep.join([app_data_folder, 'FB_data_prep.png']))
        self.data_2 = Image.open(os.sep.join([app_data_folder, 'CBOE_data_prep.png']))
        self.results1 = Image.open(os.sep.join([app_data_folder, 'FB_arima_results.png']))
        self.results2 = Image.open(os.sep.join([app_data_folder, 'FB_prophet_results.png']))
        self.results31 = Image.open(os.sep.join([app_data_folder, 'FB_lstm1_loss.png']))
        self.results32 = Image.open(os.sep.join([app_data_folder, 'FB_lstm1_results.png']))
        self.results41 = Image.open(os.sep.join([app_data_folder, 'FB_lstm2_loss.png']))
        self.results42 = Image.open(os.sep.join([app_data_folder, 'FB_lstm2_results.png']))

    def run_app(self):
        # primary app call will run everything contained in frame()
        self.frame()

    def frame(self):
        # place run_main components of page here, add more as necessary
        self.title()
        self.body()
        self.footer()

    def title(self):
        # execute st calls for title section
        st.title('Experimenting With Stock Price Prediction - Presentation Mode')

    def body(self):
        # execute st calls for body section
        # a header for this section
        sub_title = 'Application Experiment for Advanced Deep Learning'
        st.markdown(f"<h3 style='text-align: center; color: black;font-family:courier;'>{sub_title}</h3>", unsafe_allow_html=True)
        # display some overview graphs
        my_expander = st.beta_expander("Project Methods and Results (Click to Hide or Show)", expanded=False)
        with my_expander:
            st.markdown("<h2 style='text-align: center; color: black;'> * * * </h2>", unsafe_allow_html=True)
            narrative_introduction()

            st.markdown("<h2 style='text-align: center; color: black;'> * * * </h2>", unsafe_allow_html=True)
            st.header('Model Overview')
            lstm_math()

            st.markdown("<h2 style='text-align: center; color: black;'> - - </h2>", unsafe_allow_html=True)
            arima_math()

            st.markdown("<h2 style='text-align: center; color: black;'> - - </h2>", unsafe_allow_html=True)
            prophet_math()

            st.markdown("<h2 style='text-align: center; color: black;'> * * * </h2>", unsafe_allow_html=True)
            st.header('Data Preparation')
            d_col1, d_col2 = st.beta_columns(2)
            d_col1.write('Scale prices and interpolate sentiment scores with spline.')
            d_col1.image(self.data_1)
            d_col2.write('Data with varying degrees of missing values and date ranges.')
            d_col2.image(self.data_2)

            st.markdown("<h2 style='text-align: center; color: black;'> * * * </h2>", unsafe_allow_html=True)
            st.header('Data Sequences')
            st.write('Sliding Sequences of Prices')
            st.image(self.seq_1)
            st.write('Sliding Sequences of Prices and Sentiment Data')
            st.image(self.seq_2)

            st.markdown("<h2 style='text-align: center; color: black;'> * * * </h2>", unsafe_allow_html=True)
            st.header('Results')
            st.write('Baseline Experiments')
            res1 = st.beta_columns(2)
            res1[0].write('ARIMA train-predict results.')
            res1[0].image(self.results1)
            res1[1].write('FB Prophet train-predict results.')
            res1[1].image(self.results2)

            st.write('LSTM Experiments')
            res2 = st.beta_columns(2)
            res2[0].write('LSTM 1 - Stock Price Only')
            res2[0].image(self.results31)
            res2[0].image(self.results32)
            res2[1].write('LSTM 2 - Stock Price With Sentiment')
            res2[1].image(self.results41)
            res2[1].image(self.results42)

            st.markdown("<h2 style='text-align: center; color: black;'> * * * </h2>", unsafe_allow_html=True)
            st.header('LSTM Notes')
            st.write('Set default epochs to 100 with multiple early stopping conditions.')
            st.write('High variance in stock prices make prediction accuracy difficult.')

            st.markdown("<h2 style='text-align: center; color: black;'> * * * </h2>", unsafe_allow_html=True)
            st.header('Analysis of Price and Sentiment Variance')
            df = pd.read_csv(self.sample_chart)
            make_scatter(df=df, title='High-level Results for Error by Sentiment and Price Variance')
            make_histogram(filename=self.sample_chart)

            st.markdown("<h2 style='text-align: center; color: black;'> * * * </h2>", unsafe_allow_html=True)
            narrative_conclusion()
            st.markdown("<h2 style='text-align: center; color: black;'> * * * </h2>", unsafe_allow_html=True)

        # makes a sidebar selection in index
        experiment_mode = exp_mode()

        debug_type = debug_mode()

        demo_run_mode = None

        tickers = []

        if experiment_mode == 'demo':
            tickers = ['Amazon']
            run_modes_selection = default_runs()
            st.write('Default Run Modes set to:', run_modes_selection)
            if run_modes_selection == 'Baseline (ARIMA and FB Prophet)':
                demo_run_mode = ['arima', 'prophet']
            elif run_modes_selection == 'Featured (LSTMs)':
                demo_run_mode = ['lstm1', 'lstm2']
        else:
            # dummy data directories and paths
            try:
                historical_news_filename = 'news.json'
                class_data_folder = os.sep.join([os.environ['PWD'], 'data', 'class_data'])
                historical_news_path = os.sep.join([class_data_folder, historical_news_filename])
                tickers = get_stock_tickers(historical_news_path)
            except:
                pass

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
                    results_df = run_main(experiment_mode=experiment_mode, tickers=tickers, debug_mode=debug_type, demo_run_mode=demo_run_mode)

                else:
                    results_df = run_main(experiment_mode=experiment_mode, tickers=tickers, debug_mode=debug_type)

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
