import logging
import os
from datetime import date

import dvc.api
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Connection, create_engine

# from src.data_utils.utils import load_params
from src.data_utils.wavelet import wavelet_denoising2

TICKERS = {
    'DAX40': '6374',
    'NQ100': '16917',
    'SP500': '872703',
}

load_dotenv()
questdb_url = os.getenv('QUESTDB_URL')
questdb_user = os.getenv('QUESTDB_USER')
questdb_password = os.getenv('QUESTDB_PASSWORD')


QUERY_TEMPLATE = """SELECT timestamp as date, open as Open, high as High, low as Low, close as Close FROM %(table)s where timestamp>=%(start_date)s and timestamp<=%(end_date)s"""

def load_data_from_questdb(params: dict, connection: Connection, logger: logging.Logger) -> dict:
    """Load data from QuestDB."""
    try:
        data = {}
        for i in [params['index_barrier']] + params['indexes_higher']:
            # query = "SELECT timestamp as date, open as Open, high as High, low as Low, close as Close FROM %(table)s WHERE timestamp > '2026-03-01';"
            query = QUERY_TEMPLATE
            query_params = {
                "table" : params['table_name'].format(ticker=TICKERS[params['ticker']], timeframe=params['timeframes'][i].upper()),
                "start_date" : params['start_date'],
                "end_date" : params['end_date']
            }
            print(query % query_params)
            data[i] = pd.read_sql_query(query, con=connection, params=query_params, index_col='date', parse_dates=['date'])
            # data[i] = data[i].iloc[:-1]     # remove last row of unfinished candle
            print(data[i])
        return data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse source data: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading source data: %s', e)
        raise


def load_data(params: dict, logger: logging.Logger) -> dict:
    """Load data from a parquet file."""
    try:
        data = {}
        for i in [params['index_barrier']] + params['indexes_higher']:
            file = params['file_name'].format(ticker=params['ticker'], timeframe=params['timeframes'][i])     #DAX40-10s-futures.parquet
            print(file)
            # Import the data
            data[i] = pd.read_parquet(os.path.join(params['data_path'], file))
            data[i].drop(columns=['volume'], inplace=True)
            data[i]['date']=pd.to_datetime(data[i]['date'], unit='ms', utc=True)
            data[i].set_index('date', inplace=True)
            data[i].rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'}, inplace=True)
            data[i] = data[i].loc[data[i].index.date>=date.fromisoformat(params['start_date'])]
            data[i] = data[i].loc[~((data[i].index.day==1) & (data[i].index.month==1))]              # remove 1 Jan from data
            print(data[i].head())
        return data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the parquet file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(data: dict, params: dict, logger: logging.Logger) -> dict:
    """Preprocess the data by adding Close_wavelet column, date_merge column is not added"""
    try:
        data[params['index_barrier']].loc[:,'Close_wavelet'] = wavelet_denoising2(data[params['index_barrier']]['Close'], wavelet='db6', lvl=7, clear_levels=3)

        # for i in params['indexes_higher']:
            # data[i]["date_merge"] = (
            #     data[i].index
            #     + pd.to_timedelta(params['timeframe_minutes'][i], "m")
            #     - pd.to_timedelta(params['timeframe_minutes'][params['index_base']], "m")
            # )
            # print(data[i].head())

        data[params['index_base']] = (data[params['index_barrier']].groupby(data[params['index_barrier']].index.floor(f'{params['timeframes'][params['index_base']]}in'))      #ceil
                    .agg(Open=('Open','first'),
                        High=('High','max'),
                        Low=('Low','min'),
                        Close=('Close','last'),
                        Close_wavelet=('Close_wavelet','last'),
                        high_time=(params['high_time_col'],'idxmax'),
                        low_time=(params['low_time_col'],'idxmin')
                        ))

        data[params['index_base']] = data[params['index_base']].iloc[:,0:7]
        print(data[params['index_base']].head())

        return data
    except Exception as e:
        logger.error('Unexpected error occurred while preprocessing the data: %s', e)
        raise


def save_data(data: dict, params: dict, logger: logging.Logger) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        
        # Create the data/raw directory if it does not exist
        os.makedirs(f"{params['data_path_dest']}/{params['ticker']}", exist_ok=True)

        for i in [params['index_base']] + params['indexes_higher']:
            print(f'Timeframe: {params['timeframes'][i]}')
            data[i].to_csv(os.path.join(f"{params['data_path_dest']}/{params['ticker']}", 'data_ohlc_{timeframe}.csv'.format(timeframe=params['timeframes'][i])), index=True)
        
        logger.debug('Train and test data saved to %s', f"{params['data_path_dest']}/{params['ticker']}")
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main(logger: logging.Logger):
    try:
        # Load parameters from the params.yaml in the root directory
        # params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'), logger=logger)
        # Load parameters from the params.yaml in the root directory
        params = dvc.api.params_show('params.yaml')['data_ingestion']
        print(f"Params: {params}")

        if params['data_source'] == 'questdb':
            engine = create_engine(questdb_url, connect_args={
                'user': questdb_user, 'password': questdb_password,
                "connect_timeout": 5,          # 5 seconds to connect
                "options": "-c statement_timeout=10000"  # 10 seconds execution limit
            })
            with engine.connect() as connection:
                src_data = load_data_from_questdb(params, connection, logger)
            engine.dispose()
        elif params['data_source'] == 'parquet':
            src_data = load_data(params, logger)
        else:
            logger.error('Invalid data source: %s', params['data_source'])
            raise ValueError(f'Invalid data source: {params["data_source"]}')

        # Preprocess the data
        print("Data preprocesssing...")
        data = preprocess_data(src_data, params, logger)

        # unique_dates, unique_weekdates, mondays_indexes = get_dates(data, params['data_ingestion']['index_base'])
        # Save the data
        print("Data saving...")
        save_data(data, params, logger)

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':

    # Logging configuration
    logger = logging.getLogger('data_ingestion')
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('errors.log')
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    main(logger)
