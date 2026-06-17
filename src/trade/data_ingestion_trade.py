import pandas as pd
import os
import logging
from src.data_utils.static_features import static_features
from src.data_utils.utils import get_dates
from sqlalchemy import Connection, create_engine
from dotenv import load_dotenv
import dvc.api

TICKERS = {
    'DAX40': '6374',
    'NQ100': '16917',
}

load_dotenv()
questdb_url = os.getenv('QUESTDB_URL')
questdb_user = os.getenv('QUESTDB_USER')
questdb_password = os.getenv('QUESTDB_PASSWORD')


def load_data_from_questdb(params: dict, connection: Connection, logger: logging.Logger) -> dict:
    """Load data from QuestDB."""
    try:
        data = {}
        for i in params['data_ingestion']['indexes_higher'] + [params['data_ingestion']['index_base']]:
            query = "SELECT timestamp as date, open as Open, high as High, low as Low, close as Close FROM %(table)s WHERE timestamp > '2026-01-01';"
            data[i] = pd.read_sql_query(query,
            con=connection,
            index_col='date',
            parse_dates=['date'],
            params={"table" : params['data_ingestion_trade']['table_name'].format(ticker=TICKERS[params['data_ingestion_trade']['ticker']], timeframe=params['data_ingestion']['timeframes'][i].upper())}
            )
            print(data[i].head())
        return data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the parquet file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(data: dict, params: dict, logger: logging.Logger) -> dict:
    """Preprocess the data by adding date_merge column and static features"""
    try:

        for i in params['data_ingestion']['indexes_higher']:
            data[i]["date_merge"] = (
                data[i].index
                + pd.to_timedelta(params['data_ingestion']['timeframe_minutes'][i], "m")
                - pd.to_timedelta(params['data_ingestion']['timeframe_minutes'][params['data_ingestion']['index_base']], "m")
            )
            print(data[i].head())

        unique_dates, unique_weekdates, mondays_indexes = get_dates(data, params['data_ingestion']['index_base'])

        for i in params['data_ingestion']['indexes_higher'] + [params['data_ingestion']['index_base']]:
            data[i] = static_features(data[i], unique_weekdates, params['data_ingestion']['timeframe_scalers'][i], high_col="High", low_col="Low", open_col="Open", close_col="Close")
            if i != params['data_ingestion']['index_base']:
                data[i].drop(columns=['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'], inplace=True)

        print(data[params['data_ingestion']['index_base']].head())

        return data
    except Exception as e:
        logger.error('Unexpected error occurred while preprocessing the data: %s', e)
        raise


def save_data(data: dict, params: dict, data_path: str, logger: logging.Logger) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        
        # Create the data/raw directory if it does not exist
        os.makedirs(data_path, exist_ok=True)
        
        # Save the train and test data
        data[params['data_ingestion']['index_base']].to_csv(f'{data_path}/questdb_static_features_{params['data_ingestion']['timeframes'][params['data_ingestion']['index_base']]}.csv')

        for i in params['data_ingestion']['indexes_higher']:
            print(f'Timeframe: {params['data_ingestion']['timeframes'][i]}')
            data[i].to_csv(f'{data_path}/questdb_static_features_{params['data_ingestion']['timeframes'][i]}.csv')
        
        logger.debug('Train and test data saved to %s', data_path)
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
        save_data(data, params, params['data_preprocessing']['data_path_dest'], logger)

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    # Logging configuration
    logger = logging.getLogger('trade')
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('trade_errors.log')
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    main(logger)
