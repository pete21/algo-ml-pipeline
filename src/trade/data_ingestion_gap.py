import pandas as pd
import os
import logging
import requests
from sqlalchemy import Connection, create_engine, text
from dotenv import load_dotenv
import dvc.api

FETCH_URLS = {                                                                      # SELECT quoteid,marketid FROM `marketquotes` WHERE quoteid in (6374,16917,872703);
    'DAX40': 'https://charts.finsatechnology.com/data/minute/17068/mid',
    'NQ100': 'https://charts.finsatechnology.com/data/minute/20190/mid',
    'SP500': 'https://charts.finsatechnology.com/data/minute/67995/mid',
}

NUM_ROWS_TO_FETCH = 1440

TICKERS = {
    'DAX40': '6374',
    'NQ100': '16917',
    'SP500': '872703',
}

load_dotenv()
questdb_url = os.getenv('QUESTDB_URL')
questdb_user = os.getenv('QUESTDB_USER')
questdb_password = os.getenv('QUESTDB_PASSWORD')


def load_data_from_url(params: dict, logger: logging.Logger) -> dict:
    """Load data from URL"""
    try:
        fetch_url=FETCH_URLS[params['data_ingestion_trade']['ticker']]

        response = requests.get(fetch_url, params={'l': NUM_ROWS_TO_FETCH})
        data = response.json()["data"]
        data_split_rows_by_comma = [r.split(',') for r in data]
        df = pd.DataFrame(data_split_rows_by_comma)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'vol']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def save_data(data: pd.DataFrame, connection: Connection, params: dict, logger: logging.Logger) -> None:
    """Save gap data. to questdb table GAPS_<ticker>"""
    try:

        # truncate table
        table_name = f"GAPS_{TICKERS[params['data_ingestion_trade']['ticker']]}_OHLC_1M"
        # connection.execute(text(f"TRUNCATE TABLE {table_name}"))

        # insert data
        # num_rows_inserted = data.to_sql(table_name, con=connection, if_exists='append', index=False, method='multi')
        # print(f"Number of rows inserted: {num_rows_inserted}")
        for row in data.iloc[::-1].iterrows():                          # .iloc[:0:-1] - skips last unfinished candle
            if row[0].weekday()<=4:
                print("row[0]:", row[0].strftime("%Y-%m-%dT%H:%M:%S"), row[1].open, row[1].high, row[1].low, row[1].close, row[1].vol)
                # print("row[1].open:", row[1].open)
                # print("row[1].high:", row[1].high)
                # print("row[1].low:", row[1].low)
                # print("row[1].close:", row[1].close)
                # print("row[1].vol:", row[1].vol)
                connection.execute(text(f"INSERT INTO {table_name} (timestamp, open, high, low, close, vol) VALUES (to_date(:timestamp, 'yyyy-MM-ddTHH:mm:ss'), :open, :high, :low, :close, :vol)"), {
                    'timestamp': row[0].strftime("%Y-%m-%dT%H:%M:%S"),
                    'open': row[1].open,
                    'high': row[1].high,
                    'low': row[1].low,
                    'close': row[1].close,
                    'vol': row[1].vol
                })
        connection.commit()
        print(f"Rows inserted.")

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

        data = load_data_from_url(params, logger)
        print(data.head())
        data.to_csv('gaps_data_1m.csv')

        with engine.connect() as connection:
            save_data(data, connection, params, logger)
        engine.dispose()

    except Exception as e:
        logger.error('Failed to complete the gaps ingestion process: %s', e)
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
