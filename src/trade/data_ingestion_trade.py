import logging
import os

import dvc.api
import pandas as pd
import pytz
from dotenv import load_dotenv
from sqlalchemy import Connection, create_engine

from src.data_utils.static_features import static_features

TICKERS = {
    'DAX40': '6374',
    'NQ100': '16917',
    'SP500': '872703',
}

load_dotenv()
questdb_url = os.getenv('QUESTDB_URL')
questdb_user = os.getenv('QUESTDB_USER')
questdb_password = os.getenv('QUESTDB_PASSWORD')


QUERY_TEMPLATE = """SELECT timestamp as date, open as Open, high as High, low as Low, close as Close FROM %(table)s where timestamp>=%(start_date)s
UNION
select timestamp, first(open), max(high), min(low), last(close) from (
SELECT timestamp, first(open) as open, max(high) as high, min(low) as low, first(close) as close FROM
(
SELECT timestamp, open, high, low, close FROM %(gaps_table_1m)s where timestamp > (select max(timestamp) FROM  %(table_1m)s)
UNION
SELECT timestamp, open, high, low, close FROM %(tickstream_table_1m)s where timestamp > (select max(timestamp) FROM %(table_1m)s)
) group by timestamp order by timestamp) sample by """


def load_data_from_questdb(params: dict, connection: Connection, logger: logging.Logger) -> dict:
    """Load data from QuestDB."""
    try:
        data = {}
        for i in params['data_ingestion_trade']['indexes_higher'] + [params['data_ingestion_trade']['index_base']]:
            query = QUERY_TEMPLATE + params['data_ingestion_trade']['timeframes'][i].lower()
            query_params = {
                "table" : params['data_ingestion_trade']['table_name'].format(ticker=TICKERS[params['data_ingestion_trade']['ticker']], timeframe=params['data_ingestion_trade']['timeframes'][i].upper()),
                "table_1m" : params['data_ingestion_trade']['table_name'].format(ticker=TICKERS[params['data_ingestion_trade']['ticker']], timeframe='1M'),
                "tickstream_table_1m" : params['data_ingestion_trade']['tickstream_table_name'].format(ticker=TICKERS[params['data_ingestion_trade']['ticker']], timeframe='1M'),
                "gaps_table_1m" : params['data_ingestion_trade']['gaps_table_name'].format(ticker=TICKERS[params['data_ingestion_trade']['ticker']], timeframe='1M'),
                "start_date" : params['data_ingestion_trade']['start_date'],
            }
            # print(f"Query: {query % query_params}")
            data[i] = pd.read_sql_query(query, con=connection, params=query_params, index_col='date', parse_dates=['date'])
            data[i] = data[i].iloc[:-1]     # remove last row of unfinished candle
            # print(data[i].tail())
        return data

    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(data: dict, params: dict, logger: logging.Logger) -> dict:
    """Preprocess the data by adding date_merge column and static features"""
    try:

        local_timezone = pytz.timezone(params['data_ingestion_trade']['local_timezone'])

        for i in params['data_ingestion_trade']['indexes_higher'] + [params['data_ingestion_trade']['index_base']]:
            data[i]['local_date'] = data[i].index.tz_localize('UTC').tz_convert(local_timezone)
            data[i] = static_features(data[i], params['data_ingestion_trade']['timeframe_scalers'][i], high_col="High", low_col="Low", open_col="Open", close_col="Close")
            
            data[i].drop(columns=['local_date'], inplace=True)
            if i != params['data_ingestion_trade']['index_base']:
                data[i].drop(columns=['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'minute_of_day'], inplace=True)

            # print(data[i].tail())

        return data
    except Exception as e:
        logger.error('Unexpected error occurred while preprocessing the data: %s', e)
        raise


def save_data(data: dict, params: dict, data_path: str, logger: logging.Logger) -> None:
    """Save the Questdb static features data, creating the data folder if it doesn't exist."""
    try:
        
        # Create the data/raw directory if it does not exist
        os.makedirs(data_path, exist_ok=True)
        print("Saving data...")
        for i in params['data_ingestion_trade']['indexes_higher'] + [params['data_ingestion_trade']['index_base']]:
            print(f'Timeframe: {params['data_ingestion_trade']['timeframes'][i]}')
            data[i].to_csv(f'{data_path}/questdb_static_features_{params['data_ingestion_trade']['timeframes'][i]}.csv')
        
        logger.debug('Questdb static features data saved to %s', data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main(logger: logging.Logger):
    try:
        # Load parameters from the params.yaml in the root directory
        params = dvc.api.params_show('params.yaml')

        engine = create_engine(questdb_url, connect_args={
            'user': questdb_user, 'password': questdb_password,
            "connect_timeout": 5,          # 5 seconds to connect
            "options": "-c statement_timeout=10000"  # 10 seconds execution limit
        })
        with engine.connect() as connection:
            src_data = load_data_from_questdb(params, connection, logger)
        engine.dispose()

        # Load data from the specified URL
        data = preprocess_data(src_data, params, logger)

        # Save the data
        save_data(data, params, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'), logger)

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    # Logging configuration
    logger = logging.getLogger('trade')
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('trade_agent_log.log')
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    main(logger)
