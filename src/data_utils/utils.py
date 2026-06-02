from yaml import YAMLError, safe_load
import logging
import numpy as np

def load_params(params_path: str, logger: logging.Logger) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def get_dates(data: dict, index: int) -> tuple[list, list, list]:
    unique_dates = np.unique(data[index].index.date)
    unique_weekdates = []
    for d in unique_dates:
        if d.weekday()<5:
            unique_weekdates.append(d)
    print(len(unique_dates), len(unique_weekdates))

    mondays_indexes = [i for i, n in enumerate(unique_dates) if n.weekday() == 0]
    print(mondays_indexes)
    num_mondays = sum(1 for i in unique_dates if i.weekday() == 0)
    print(num_mondays)
    return unique_dates, unique_weekdates, mondays_indexes