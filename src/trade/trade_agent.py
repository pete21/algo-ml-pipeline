import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaConsumer

from src.trade.data_ingestion_trade import main as data_ingestion_trade_main
from src.trade.db_utils import (
    get_mysql_connection,
    get_open_orders,
    insert_db_order,
    insert_predictions,
    update_order_from_kafka_event,
    update_order_from_marketbroker_response,
)
from src.trade.model_inference_trade import main as model_inference_trade_main
from src.trade.utils_marketbroker import (
    cancel_pending_order,
    get_current_price,
    submit_limit_order,
)
from src.trade.utils_model_serving import fetch_model_info

load_dotenv()

EXECUTE_ORDER_THRESHOLD = 1
ORDER_PRICE_SHIFT = 0.0002

INTERVAL_MINUTES = 5
SCHEDULE_OFFSET_SECONDS = 5

CANCEL_PENDING_ORDER_OLDER_THAN_MINUTES = INTERVAL_MINUTES+1 # maximum lifetime of a pending order, then it is is cancelled by the trade agent

TICKER = 'DAX40'

ORDER_MODE_MARKET = 0
ORDER_MODE_LIMIT = 1
ORDER_STAKE = 0.5
KAFKA_URL = os.getenv('KAFKA_URL', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'MARKETBROKER.LOCAL.TRANSACTIONS_TOPIC')
KAFKA_CONSUMER_GROUP = os.getenv('KAFKA_CONSUMER_GROUP', 'trade-agent-transactions')


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
    timeseries_datetime date_val
    ticker TICKER (DAX40)
    side    1 (buy) or -1 (sell)
    price    result['Close'][date_val]
    amount   0.1
    tp    buy: price * (1 + tp) · sell: price * (1 - tp)
    sl    buy: price * (1 - sl) · sell: price * (1 + sl)
    status   1
    created_at    DB default (CURRENT_TIMESTAMP)

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

    # Get the current price of the ticker
    current_price = get_current_price(ticker=ticker, side='ask' if side == 'buy' else 'bid', logger=logger)
    if current_price is not None:
        price = (price + current_price) / 2                     # Adjust the price to the average of the current market price and the last closed candle price

    price = round(price * (1 - order_side * ORDER_PRICE_SHIFT), 2)

    inserted_id = insert_db_order(connection, result, idx, ticker, order_side, price, stake, tp, sl, logger)

    if inserted_id is not None:
        print(f"Order inserted for {result.index[idx]}")
        logger.info("Order inserted for %s", result.index[idx])

        try:
            response = submit_limit_order(
                logger=logger,
                ticker=ticker,
                order_mode=ORDER_MODE_LIMIT,
                direction=order_side,
                stake=stake,
                price=price,
                stop_order_price=sl,
                limit_order_price=tp,
                trailing_point=False,
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
                update_order_from_marketbroker_response(connection, inserted_id, order_id, active, 'PENDING', error=None, logger=logger)
            else:
                update_order_from_marketbroker_response(connection, inserted_id, 0, False, 'ERRORED', error={'info': json.dumps(response)}, logger=logger)

            logger.info("Limit order request updated for prediction %s: %s", result.index[idx], response)
            print(f"Limit order request updated for prediction {result.index[idx]}: {response}")

        except Exception as exc:
            logger.error("Error submitting limit order request to market API for prediction %s: %s", result.index[idx], exc)
            print(f"Error submitting limit order request to market API for prediction {result.index[idx]}: {exc}")
            update_order_from_marketbroker_response(connection, inserted_id, 0, False, 'REJECTED', error={'info': str(exc)}, logger=logger)

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


def _sleep_until_next_cycle(logger: logging.Logger, interval_minutes: int = 5, schedule_offset_seconds: int = 5) -> datetime:
    """Block until the next scheduled cycle time."""
    now = datetime.now()
    target = _next_scheduled_time(now=now, interval_minutes=interval_minutes, schedule_offset_seconds=schedule_offset_seconds)
    sleep_seconds = (target - now).total_seconds()
    if sleep_seconds > 0:
        logger.info(
            "Sleeping %.1f seconds until next cycle at %s",
            sleep_seconds,
            target.strftime("%Y-%m-%d %H:%M:%S"),
        )
        time.sleep(sleep_seconds)
    return target

def _next_scheduled_time(now: datetime | None = None, interval_minutes: int = 5, schedule_offset_seconds: int = 5) -> datetime:
    """Return the next run time aligned to 5-minute boundaries plus offset (e.g. :05:05, :10:05)."""
    now = now or datetime.now()
    current = now.replace(microsecond=0)
    seconds_into_hour = current.minute * 60 + current.second
    interval_seconds = interval_minutes * 60
    slot_start = (seconds_into_hour // interval_seconds) * interval_seconds
    candidate = current.replace(minute=0, second=0) + timedelta(
        seconds=slot_start + schedule_offset_seconds
    )
    if candidate < now:
        candidate += timedelta(seconds=interval_seconds)
    return candidate



def run_cycle() -> None:
    """Run a single cycle of the trade agent."""
    model_info = fetch_model_info(logger=logger)
    registered_model_name = model_info['registered_model_name']
    model_version = model_info['model_version']

    logger.info("Starting data ingestion...")
    data_ingestion_trade_main(logger)
    logger.info("Data ingestion completed.")
    result = model_inference_trade_main(logger)
    logger.info("Model inference completed.")

    if result is None or result.empty:
        logger.warning("No inference results to insert.")
        return

    mysql_connection = get_mysql_connection()
    try:
        inserted = insert_predictions(mysql_connection, result.y_pred, result.raw_prediction, TICKER, registered_model_name, model_version, logger)

        # Check if there are any open orders for the ticker
        open_orders = get_open_orders(mysql_connection, TICKER)
        if open_orders:

            # TODO: Check if the pending orders are older than CANCEL_PENDING_ORDER_OLDER_THAN_MINUTES minutes and cancel them if they are
            pending_orders = [order for order in open_orders if order['status'] == 1]
            print(f"Number of pending orders: {len(pending_orders)}")
            for order in pending_orders:
                if order['created_at'].timestamp() < datetime.now().timestamp()-7200 - CANCEL_PENDING_ORDER_OLDER_THAN_MINUTES * 60:
                    cancelled = cancel_pending_order(order['order_id'], logger=logger)
                    if cancelled:
                        print(f"Updating cancelled order {order['id']} because it is older than {CANCEL_PENDING_ORDER_OLDER_THAN_MINUTES} minutes")
                        logger.info("Updating cancelled order %s", order['id'])
                        update_order_from_marketbroker_response(mysql_connection, order['id'], order['order_id'], True, 'CANCEL_PENDING', error={'info': f"Order is {round(datetime.now().timestamp()-7200-order['created_at'].timestamp())} seconds old (>{CANCEL_PENDING_ORDER_OLDER_THAN_MINUTES} minutes permitted)"}, logger=logger)
                        logger.info("Cancelled pending order %s because it is older than %d minutes", order['order_id'], CANCEL_PENDING_ORDER_OLDER_THAN_MINUTES)
                    else:
                        logger.error("Failed to cancel pending order %s", order['order_id'])
                        print(f"Failed to cancel pending order {order['order_id']}")

            logger.info("Skipping order processing: There are pending or opened order(s): %s", open_orders)
            print(f"Skipping order processing: There are pending or opened order(s): {open_orders}")
            return

        inserted_id = inserted[-1]
        idx = -1
        if inserted_id is None:
            logger.warning("No predictions inserted.")
            return

        orders_inserted_ids = []
        if abs(result.y_pred.iloc[idx]) > EXECUTE_ORDER_THRESHOLD and result.index[idx].timestamp() < datetime.now().timestamp()-7200-600:
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
    """Main function to run the trade agent."""
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
        scheduled_at = _sleep_until_next_cycle(logger=logger, interval_minutes=INTERVAL_MINUTES, schedule_offset_seconds=SCHEDULE_OFFSET_SECONDS)
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
