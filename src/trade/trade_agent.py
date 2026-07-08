import os
import logging
import threading
import time
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaConsumer

from src.trade.data_ingestion_trade import main as data_ingestion_trade_main
from src.trade.db_utils import get_mysql_connection, get_open_orders, insert_db_order, insert_predictions, sleep_until_next_cycle, update_order_from_marketbroker_response, update_order_from_kafka_event
from src.trade.model_inference_trade import main as model_inference_trade_main

load_dotenv()
EXECUTE_ORDER_THRESHOLD = 0.5
ORDER_PRICE_SHIFT = 0.0002

INTERVAL_MINUTES = 5
SCHEDULE_OFFSET_SECONDS = 5

TICKER = 'DAX40'
ORDER_API_URL = os.getenv('ORDER_API_URL', 'http://localhost:8080/orders')
ORDER_MODE_MARKET = 0
ORDER_MODE_LIMIT = 1
ORDER_STAKE = 0.5
KAFKA_URL = os.getenv('KAFKA_URL', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'MARKETBROKER.LOCAL.TRANSACTIONS_TOPIC')
KAFKA_CONSUMER_GROUP = os.getenv('KAFKA_CONSUMER_GROUP', 'trade-agent-transactions')


# Example market/quote IDs keyed by ticker symbol.
TICKER_MARKET_MAP: dict[str, dict[str, int]] = {
    'DAX40': {'marketId': 17068, 'quoteId': 6374},
    'NQ100': {'marketId': 20190, 'quoteId': 16917},
    'SP500': {'marketId': 67995, 'quoteId': 872703},
    'Bitcoin': {'marketId': 67476, 'quoteId': 870964},
}


# Logging configuration
logger = logging.getLogger('trade')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('trade_agent_log.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def submit_limit_order(
    ticker: str,
    order_mode: int,
    direction: int,
    stake: float,
    price: float = 0.0,
    stop_order_price: float = 0.0,
    limit_order_price: float = 0.0,
    trailing_point: bool = False,
    url: str = ORDER_API_URL
) -> dict:
    """Submit a limit order to the external order API."""
    if ticker not in TICKER_MARKET_MAP:
        raise ValueError(f"No market mapping configured for ticker: {ticker}")

    market = TICKER_MARKET_MAP[ticker]
    payload = {
        'marketId': market['marketId'],
        'quoteId': market['quoteId'],
        'price': float(price),
        'stake': float(stake),
        'direction': direction,
        'orderMode': order_mode,
        'limitOrderPrice': float(limit_order_price),
        'stopOrderPrice': float(stop_order_price),
        'trailingPoint': trailing_point,
    }
    print(payload)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode('utf-8')
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode('utf-8')
        raise RuntimeError(
            f"Limit order request failed with status {exc.code}: {error_body}"
        ) from exc



def insert_db_marketbroker_order(
    connection: mysql.connector.MySQLConnection,
    result: pd.DataFrame,
    idx: int,
    side: str,
    ticker: str,
) -> int:
    """Insert a single order into the orders table.

    Column	Value
    inference_id    0
    signal_timestamp    DB default (CURRENT_TIMESTAMP)
    timeseries_datetime date_val
    ticker TICKER (DAX40)
    side    1 (buy) or -1 (sell)
    price    result['Close'][date_val]
    amount   0.1
    tp    buy: price * (1 + tp) · sell: price * (1 - tp)
    sl    buy: price * (1 - sl) · sell: price * (1 + sl)
    status   1

    """
    price = float(result['Close'].iloc[idx])
    tp_factor = float(result['tp'].iloc[idx])
    sl_factor = float(result['sl'].iloc[idx])
    stake = ORDER_STAKE

    if side == 'buy':
        order_side = 1
        sl = round(price * (1 - sl_factor), 2)
        tp = round(price * (1 + tp_factor), 2)
    elif side == 'sell':
        order_side = -1
        sl = round(price * (1 + sl_factor), 2)
        tp = round(price * (1 - tp_factor), 2)
    else:
        raise ValueError(f"Invalid side: {side}")

    price = round(price * (1 - order_side * ORDER_PRICE_SHIFT), 2)

    inserted_id = insert_db_order(connection, result, idx, ticker, order_side, price, stake, tp, sl)

    if inserted_id is not None:
        print(f"Order inserted for {result.index[idx]}")
        logger.info("Order inserted for %s", result.index[idx])

        try:
            response = submit_limit_order(
                ticker=ticker,
                order_mode=ORDER_MODE_LIMIT,
                direction=order_side,
                stake=stake,
                price=price,
                stop_order_price=sl,
                limit_order_price=tp,
                trailing_point=False,
                url=ORDER_API_URL,
            )
# Request body:
# {
#   "marketId": 17068,
#   "quoteId": 6374,
#   "price": 24830,
#   "stake": 1,
#   "direction": 1,
#   "orderMode": 1,
#   "limitOrderPrice": 24890,
#   "stopOrderPrice": 24760,
#   "trailingPoint": false
# }
# Response body:
# {
#     "orderId": 26823511,
#     "marketId": 0,
#     "quoteId": 0,
#     "price": 24830.0,
#     "stake": 1.0,
#     "direction": 1,
#     "limitOrderPrice": 24890.0,
#     "stopOrderPrice": 24760.0,
#     "trailingPoint": false,
#     "message": "",
#     "positionId": 0,
#     "active": true,
#     "status": 0
# }
            if response['status'] == 0:
                active = bool(response['active'])
                order_id = int(response['orderId'])
                update_order_from_marketbroker_response(connection, inserted_id, order_id, active, 'PENDING', error=None)
            else:
                update_order_from_marketbroker_response(connection, inserted_id, 0, False, 'ERRORED', error={'info': json.dumps(response)})

            logger.info("Limit order request updated for prediction %s: %s", result.index[idx], response)
            print(f"Limit order request updated for prediction {result.index[idx]}: {response}")

        except Exception as exc:
            logger.error("Error submitting limit order request to market API for prediction %s: %s", result.index[idx], exc)
            print(f"Error submitting limit order request to market API for prediction {result.index[idx]}: {exc}")
            update_order_from_marketbroker_response(connection, inserted_id, 0, False, 'ERRORED', error={'info': str(exc)})

    else:
        print(f"Order could not be inserted into database for {result.index[idx]}")
        logger.warning("Order could not be inserted into database for %s", result.index[idx])
    return inserted_id


def kafka_listener_loop(stop_event: threading.Event) -> None:
    """Consume transaction events from Kafka and update order statuses."""
    logger.info(
        'Starting Kafka transaction listener on %s topic %s',
        KAFKA_URL,
        KAFKA_TOPIC,
    )
    time.sleep(5)
    while not stop_event.is_set():
        consumer = None
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=[server.strip() for server in KAFKA_URL.split(',') if server.strip()],
                group_id=KAFKA_CONSUMER_GROUP,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                value_deserializer=json.loads,
            )
            for message in consumer:
                if stop_event.is_set():
                    break
                try:
                    connection = get_mysql_connection()
                    try:
                        update_order_from_kafka_event(connection, message.value, logger)
                    finally:
                        connection.close()
                except Exception as exc:
                    logger.error('Failed to process transaction event: %s', exc)
                    if stop_event.wait(5):
                        break
        except Exception as exc:
            logger.error('Kafka consumer error: %s', exc)
            if stop_event.wait(5):
                break
        finally:
            if consumer is not None:
                consumer.close()


def run_cycle() -> None:
    logger.info("Starting data ingestion...")
    data_ingestion_trade_main(logger)
    logger.info("Data ingestion completed. Starting model inference...")
    result = model_inference_trade_main(logger)
    logger.info("Model inference completed.")

    if result is None or result.empty:
        logger.warning("No inference results to insert.")
        return

    registered_model_name = os.getenv('MODEL_NAME') + '_' + os.getenv('MODEL_DATE')
    model_version = 0
    mysql_connection = get_mysql_connection()
    try:
        inserted = insert_predictions(mysql_connection, result.y_pred, TICKER, registered_model_name, model_version, logger)

        # Check if there are any open orders for the ticker
        open_orders = get_open_orders(mysql_connection, TICKER)
        if open_orders:
            logger.info("Skipping order processing: There are existing open order(s): %s", open_orders)
            print(f"Skipping order processing: There are existing open order(s): {open_orders}")
            return

        orders_inserted_ids = []
        # for idx, inserted_id in enumerate(inserted):

        inserted_id = inserted[-1]
        idx = -1
        if inserted_id is not None:

            if abs(result.y_pred.iloc[idx]) > EXECUTE_ORDER_THRESHOLD and result.index[idx].tz_localize(timezone.utc) < datetime.now(tz=timezone.utc) - timedelta(minutes=10):
                print(f"Skipping order insertion for {result.index[idx]} because it is older than 5 minutes")
                logger.warning("Skipping order insertion for %s because it is older than 5 minutes", result.index[idx])
            else:
                if result.y_pred.iloc[idx] > EXECUTE_ORDER_THRESHOLD:
                    orders_inserted_ids.append(insert_db_marketbroker_order(mysql_connection, result, idx, 'buy', TICKER))
                elif result.y_pred.iloc[idx] < -EXECUTE_ORDER_THRESHOLD:
                    orders_inserted_ids.append(insert_db_marketbroker_order(mysql_connection, result, idx, 'sell', TICKER))

        if len(orders_inserted_ids) > 0:
            print(f"Inserted {len(orders_inserted_ids)} new rows into orders")
            logger.info("Inserted %d new rows into orders", len(orders_inserted_ids))
    finally:
        mysql_connection.close()


########################################################
# Main function
########################################################

def main() -> None:
    logger.info(
        "Starting trade agent daemon (every %d minutes, %d seconds after the period)",
        INTERVAL_MINUTES,
        SCHEDULE_OFFSET_SECONDS,
    )
    stop_event = threading.Event()
    kafka_thread = threading.Thread(
        target=kafka_listener_loop,
        args=(stop_event,),
        name='kafka-transactions-listener',
        daemon=True,
    )
    kafka_thread.start()
    logger.info('Kafka transactions event listener thread started.')
    while True:
        scheduled_at = sleep_until_next_cycle(logger, INTERVAL_MINUTES, SCHEDULE_OFFSET_SECONDS)
        logger.info("Cycle start time: %s (scheduled: %s)", datetime.now(), scheduled_at)
        try:
            run_cycle()
            logger.info("Trade agent cycle completed successfully.")
        except Exception as e:
            logger.error("Trade agent cycle failed: %s", e)
        logger.info("Cycle end time: %s", datetime.now())


    # # on shutdown (SIGINT handler, atexit, etc.)
    # stop_event.set()
    # kafka_thread.join(timeout=10)

if __name__ == '__main__':
    main()
