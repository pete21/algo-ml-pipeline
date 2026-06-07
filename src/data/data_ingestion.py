import pandas as pd
import os
import logging
from datetime import date
from src.data_utils.utils import get_dates, load_params
from src.data_utils.wavelet import wavelet_denoising2


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

def load_data(params: dict, data_path: str) -> dict:
    """Load data from a parquet file."""
    try:
        dax_data = {}
        for i in params['data_ingestion']['indexes']:
            file = params['data_ingestion']['file_name'].format(ticker=params['data_ingestion']['ticker'], timeframe=params['data_ingestion']['timeframes'][i])     #DAX40_USDT_USDT-10s-futures.parquet
            print(file)
            # Import the data
            dax_data[i] = pd.read_parquet(os.path.join(data_path, file))
            dax_data[i].drop(columns=['volume'], inplace=True)
            dax_data[i]['date']=pd.to_datetime(dax_data[i]['date'], unit='ms', utc=True)
            dax_data[i].set_index('date', inplace=True)
            dax_data[i].rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'}, inplace=True)
            dax_data[i] = dax_data[i].loc[dax_data[i].index.date>=date(2023,4,3)]
            dax_data[i] = dax_data[i].loc[~((dax_data[i].index.day==1) & (dax_data[i].index.month==1))]              # remove 1 Jan from data
            print(dax_data[i].head())
        return dax_data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the parquet file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(dax_data: dict, params: dict) -> dict:
    """Preprocess the data by adding Close_wavelet column and date_merge column"""
    try:
        dax_data[params['data_ingestion']['index_barrier']].loc[:,'Close_wavelet'] = wavelet_denoising2(dax_data[params['data_ingestion']['index_barrier']]['Close'], wavelet='db6', lvl=8, clear_levels=3)

        for i in params['data_ingestion']['indexes_higher']:
            dax_data[i]["date_merge"] = (
                dax_data[i].index
                + pd.to_timedelta(params['data_ingestion']['timeframe_minutes'][i], "m")
                - pd.to_timedelta(params['data_ingestion']['timeframe_minutes'][params['data_ingestion']['index_base']], "m")
            )
            print(dax_data[i].head())

        dax_data[params['data_ingestion']['index_base']] = (dax_data[params['data_ingestion']['index_barrier']].groupby(dax_data[params['data_ingestion']['index_barrier']].index.floor(f'{params['data_ingestion']['timeframes'][params['data_ingestion']['index_base']]}in'))      #ceil
                    .agg(Open=('Open','first'),
                        High=('High','max'),
                        Low=('Low','min'),
                        Close=('Close','last'),
                        Close_wavelet=('Close_wavelet','last'),
                        high_time=(params['data_ingestion']['high_time_col'],'idxmax'),
                        low_time=(params['data_ingestion']['low_time_col'],'idxmin')
                        ))

        dax_data[params['data_ingestion']['index_base']] = dax_data[params['data_ingestion']['index_base']].iloc[:,0:7]
        print(dax_data[params['data_ingestion']['index_base']].head())

        return dax_data
    except Exception as e:
        logger.error('Unexpected error occurred while preprocessing the data: %s', e)
        raise


def save_data(data: dict, params: dict, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        
        # Create the data/raw directory if it does not exist
        os.makedirs(data_path, exist_ok=True)
        
        # Save the train and test data
        data[params['data_ingestion']['index_base']].to_csv(f'{data_path}/data_ohlc_features_{params['data_ingestion']['timeframes'][params['data_ingestion']['index_base']]}.csv')

        for i in params['data_ingestion']['indexes_higher']:
            print(f'Timeframe: {params['data_ingestion']['timeframes'][i]}')
            data[i].to_csv(f'{data_path}/data_ohlc_features_{params['data_ingestion']['timeframes'][i]}.csv')
        
        logger.debug('Train and test data saved to %s', data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # Load parameters from the params.yaml in the root directory
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'), logger=logger)
        
        # Load data from the specified URL
        src_data = load_data(params, params['data_ingestion']['data_path'])
        
        # Preprocess the data
        data = preprocess_data(src_data, params)

        unique_dates, unique_weekdates, mondays_indexes = get_dates(data, params['data_ingestion']['index_base'])
        
        # Save the data
        save_data(data, params, params['data_ingestion']['data_path_dest'])

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
