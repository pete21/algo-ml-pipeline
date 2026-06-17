import logging
import time
from datetime import datetime

from dotenv import load_dotenv

import data_ingestion_trade
import model_inference_trade

load_dotenv()

INTERVAL_SECONDS = 5 * 60

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


def run_cycle() -> None:
    logger.info("Starting data ingestion...")
    data_ingestion_trade.main(logger)
    logger.info("Data ingestion completed. Starting model inference...")
    model_inference_trade.main(logger)
    logger.info("Model inference completed.")


def main() -> None:
    logger.info("Starting trade agent daemon (interval: %d seconds)", INTERVAL_SECONDS)
    while True:
        cycle_start = datetime.now()
        logger.info("Cycle start time: %s", cycle_start)
        try:
            run_cycle()
            logger.info("Trade agent cycle completed successfully.")
        except Exception as e:
            logger.error("Trade agent cycle failed: %s", e)
        logger.info("Cycle end time: %s", datetime.now())
        logger.info("Sleeping for %d seconds until next cycle...", INTERVAL_SECONDS)
        time.sleep(INTERVAL_SECONDS)


if __name__ == '__main__':
    main()
