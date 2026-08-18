import logging
import os

import mysql.connector
import pandas as pd

INFERENCE_TABLE = 'inference'
ORDERS_TABLE = 'orders'
TRANSACTION_TYPE_TO_STATUS = {
    'ERRORED': -1,
    'PLANNED': 0,
    'PENDING': 1,
    'FILLED': 2,
    'CLOSED': 3,
    'REJECTED': 4,
    'CANCEL_PENDING': 5,
    'CANCELLED': 6,
}


def _parse_mysql_host_port(mysql_url: str) -> tuple[str, int]:
    if ':' in mysql_url:
        host, port = mysql_url.rsplit(':', 1)
        return host, int(port)
    return mysql_url, 3306


def get_mysql_connection() -> mysql.connector.MySQLConnection:
    mysql_host, mysql_port = _parse_mysql_host_port(os.getenv('MYSQL_URL', 'localhost:3306'))
    return mysql.connector.connect(
        host=mysql_host,
        port=mysql_port,
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DATABASE'),
    )


def insert_predictions(connection: mysql.connector.MySQLConnection, y_series: pd.Series, raw_prediction: pd.Series, ticker: str, registered_model_name: str, model_version: int, logger: logging.Logger) -> list[int]:
    """Insert inference results into MySQL, skipping rows with duplicate dates."""
    cursor = connection.cursor()
    sql = f"INSERT IGNORE INTO {INFERENCE_TABLE} (ticker, timeseries_datetime, prediction, raw_prediction, registered_model_name, model_version) VALUES (%s, %s, %s, %s, %s, %s)"
    print(sql % (ticker, pd.Timestamp(y_series.index[0]).to_pydatetime(), float(y_series.values[0]), float(raw_prediction.values[0]), registered_model_name, model_version))
    logger.info(sql % (ticker, pd.Timestamp(y_series.index[0]).to_pydatetime(), float(y_series.values[0]), float(raw_prediction.values[0]), registered_model_name, model_version))
    inserted_ids = []
    for date_val, prediction_val, raw_prediction_val in zip(y_series.index, y_series.values, raw_prediction.values):
        cursor.execute(sql, (ticker, pd.Timestamp(date_val).to_pydatetime(), float(prediction_val), float(raw_prediction_val), registered_model_name, model_version))
        if cursor.rowcount > 0:
            inserted_ids.append(cursor.lastrowid)
            print(f"Prediction inserted for {date_val}")
            logger.info("Prediction inserted for %s", date_val)
        else:
            inserted_ids.append(None)
            # print(f"No prediction inserted for {date_val}")
            # logger.warning("No prediction inserted for %s", date_val)

    connection.commit()
    cursor.close()
    return inserted_ids


def insert_db_order(connection: mysql.connector.MySQLConnection, result: pd.DataFrame, idx: int, ticker: str, order_side: int, price: float, stake: float, tp: float, sl: float, logger: logging.Logger) -> int:
    cursor = connection.cursor()
    sql = (
        f"INSERT INTO {ORDERS_TABLE} "
        "(inference_id, timeseries_datetime, ticker, side, price, amount, tp, sl, status) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    print(sql % (0, pd.Timestamp(result.index[idx]).to_pydatetime(), ticker, order_side, price, stake, tp, sl, TRANSACTION_TYPE_TO_STATUS['PLANNED']))
    logger.info(sql % (0, pd.Timestamp(result.index[idx]).to_pydatetime(), ticker, order_side, price, stake, tp, sl, TRANSACTION_TYPE_TO_STATUS['PLANNED']))
    cursor.execute(
        sql,
        (
            0,
            pd.Timestamp(result.index[idx]).to_pydatetime(),
            ticker,
            order_side,
            price,
            stake,
            tp,
            sl,
            TRANSACTION_TYPE_TO_STATUS['PLANNED'],
        ),
    )
    connection.commit()
    inserted_id = cursor.lastrowid if cursor.rowcount > 0 else None
    cursor.close()
    return inserted_id


def update_order_from_marketbroker_response(connection: mysql.connector.MySQLConnection, inserted_id: int, order_id: int, active: bool, status: str, error: dict, logger: logging.Logger) -> None:
    cursor = connection.cursor()
    sql = f"UPDATE {ORDERS_TABLE} SET active = %s, order_id = %s, status = %s, updated_at = NOW() WHERE id = %s"
    print(sql % (active, order_id, TRANSACTION_TYPE_TO_STATUS[status], inserted_id))
    logger.info(sql % (active, order_id, TRANSACTION_TYPE_TO_STATUS[status], inserted_id))
    cursor.execute(sql, (active, order_id, TRANSACTION_TYPE_TO_STATUS[status], inserted_id))
    if error is not None:
        sql = f"UPDATE {ORDERS_TABLE} SET info = %s, status = %s, updated_at = NOW() WHERE id = %s"
        print(sql % (error['info'], TRANSACTION_TYPE_TO_STATUS[status], inserted_id))
        logger.info(sql % (error['info'], TRANSACTION_TYPE_TO_STATUS[status], inserted_id))
        cursor.execute(f"UPDATE {ORDERS_TABLE} SET info = %s, status = %s, updated_at = NOW() WHERE id = %s", (error['info'], TRANSACTION_TYPE_TO_STATUS[status], inserted_id))
    connection.commit()
    cursor.close()


def get_open_orders(connection: mysql.connector.MySQLConnection, ticker: str) -> dict:
    """Get all open orders for the ticker."""
    cursor = connection.cursor()
    sql = f"SELECT * FROM {ORDERS_TABLE} WHERE ticker = %s AND status in (1,2)"
    cursor.execute(sql, (ticker,))
    orders = cursor.fetchall()
    order_dict = [dict(zip([key[0] for key in cursor.description], row)) for row in orders]
    cursor.close()
    return order_dict if order_dict else {}


def update_order_from_kafka_event(connection: mysql.connector.MySQLConnection, event: dict, logger: logging.Logger) -> None:
    """Update order from a market broker transaction event."""
    print("Kafka event:", event)
    logger.info("Kafka event:", event)
    order_id = event.get('o')
    event_type = event.get('type')
    if order_id is None or event_type is None:
        logger.warning('Transaction event missing required fields: %s', event)
        return

    status = TRANSACTION_TYPE_TO_STATUS.get(event_type)
    if status is None:
        logger.debug('Ignoring transaction event with unhandled type %s: %s', event_type, event)
        return

    position_id = event.get('p')
    price = event.get('price')
    try:
        cursor = connection.cursor()

        match event_type:
            case 'PENDING':
                sql = f'UPDATE {ORDERS_TABLE} SET status = %s, updated_at = NOW() WHERE order_id = %s and status = 0'
                print(sql % (status, int(order_id)))
                logger.info(sql % (status, int(order_id)))
                cursor.execute(sql, (status, int(order_id)))
            case 'FILLED':
                sql = f'UPDATE {ORDERS_TABLE} SET status = %s, position_id = %s, open_price = %s, updated_at = NOW() WHERE order_id = %s and status in (0,1)'
                print(sql % (status, int(position_id), float(price), int(order_id)))
                logger.info(sql % (status, int(position_id), float(price), int(order_id)))
                cursor.execute(sql, (status, int(position_id), float(price), int(order_id)))
            case 'CLOSED':
                sql = f'UPDATE {ORDERS_TABLE} SET active = 0, status = %s, close_price = %s, updated_at = NOW() WHERE order_id = %s and status = 2'
                print(sql % (status, float(price), int(order_id)))
                logger.info(sql % (status, float(price), int(order_id)))
                cursor.execute(sql, (status, float(price), int(order_id)))
            case 'CANCELLED':
                sql = f'UPDATE {ORDERS_TABLE} SET active = 0, status = %s, updated_at = NOW() WHERE order_id = %s'      # and status in (5)
                print(sql % (status, int(order_id)))
                logger.info(sql % (status, int(order_id)))
                cursor.execute(sql, (status, int(order_id)))
            case _:
                logger.warning('Ignoring transaction event with unhandled type %s: %s', event_type, event)

        connection.commit()
        if cursor.rowcount > 0:
            logger.info('Updated order %s status to %s (%s)', order_id, status, event_type)
        else:
            logger.warning('No order updated for event %s', event)
    except Exception as exc:
        logger.error('Failed to update order from event: %s', exc)
    finally:
        cursor.close()

