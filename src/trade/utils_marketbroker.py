import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

ORDER_API_URL = os.getenv('ORDER_API_URL', 'http://localhost:8080/orders')
INSTRUMENTS_API_URL = os.getenv('INSTRUMENTS_API_URL', 'http://localhost:8080/instruments')

# Example market/quote IDs keyed by ticker symbol.
TICKER_MARKET_MAP: dict[str, dict[str, int]] = {
    'DAX40': {'marketId': 17068, 'quoteId': 6374},
    'NQ100': {'marketId': 20190, 'quoteId': 16917},
    'SP500': {'marketId': 67995, 'quoteId': 872703},
    'Bitcoin': {'marketId': 67476, 'quoteId': 870964},
}


def cancel_pending_order(order_id: int, logger: logging.Logger) -> bool:
    """Cancel a pending order."""
    url = f"{ORDER_API_URL}/{order_id}"
    try:
        response = requests.delete(
            url,
        )
        return response.status_code == 200
    except requests.exceptions.RequestException as exc:
        logger.error("Order %s cancellation failed: %s", order_id, exc)
        return False


def submit_limit_order(
    logger: logging.Logger,
    ticker: str,
    order_mode: int,
    direction: int,
    stake: float,
    price: float = 0.0,
    stop_order_price: float = 0.0,
    limit_order_price: float = 0.0,
    trailing_point: bool = False,
) -> dict:
    """Submit a limit order to the external order API."""
    if ticker not in TICKER_MARKET_MAP:
        logger.error(f"No market mapping configured for ticker: {ticker}")
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

    try:
        response = requests.post(
            ORDER_API_URL,
            json=payload
        )
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Limit order request failed: %s", exc)
        raise

def close_position(ticker: str, order_id: int, direction: int, logger: logging.Logger) -> bool:
    """Close a position."""
    if ticker not in TICKER_MARKET_MAP:
        logger.error(f"No market mapping configured for ticker: {ticker}")
        raise ValueError(f"No market mapping configured for ticker: {ticker}")
    
    market = TICKER_MARKET_MAP[ticker]
    payload = {
        "marketId": market['marketId'],
        "quoteId": market['quoteId'],
        "price": 0,
        "stake": 0,
        "direction": direction,
        "orderMode": 5,
        "limitOrderPrice": 0,
        "stopOrderPrice": 0,
        "trailingPoint": False,
        "positionId": order_id
    }
    print(payload)

    try:
        response = requests.post(
            ORDER_API_URL,
            json=payload
        )
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Position with order ID %s closing failed: %s", order_id, exc)
        raise

# Query /instruments/ticks to get the current price of the ticker
def get_current_price(ticker: str, side: str, logger: logging.Logger) -> float:
    quote_id = TICKER_MARKET_MAP[ticker]['quoteId']
    if side not in ['bid', 'ask']:
        logger.error("Invalid side: %s", side)
        return None
    '''
    Example response:
    [
    { "quoteId": 6374, "bid": 24976.4, "ask": 24977.4, "time": 1784884288, "millis": 299},
    { "quoteId": 16917, "bid": 24976.5, "ask": 24977.5, "time": 1784884288, "millis": 299},
    ]
    '''
    url = f"{INSTRUMENTS_API_URL}/ticks"
    response = requests.get(url)
    if response.status_code != 200:
        logger.error("Failed to get current price for %s: %s", ticker, response.status_code)
        print(f"Failed to get current price for {ticker}: {response.status_code}")
        return None
    json_response = response.json()
    for item in json_response:
        if item['quoteId'] == quote_id:
            return item[side]
    logger.error("Failed to find quoteId %s in ticks response", quote_id)
    print(f"Failed to find quoteId {quote_id} in ticks response")
    return None

