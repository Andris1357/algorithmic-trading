from binance.client import Client
import csv
import time
from typing import Union

new_line = '\n'

api_key = 'BkFkzGa7dWD9nMXBIPsSrfXMpEbLyNmE4EqimcpyvrF4KhxfoQJTGnvkg5W6xtg0'
api_secret = 'J0vg4frGj9dJBpX3hVw8nfDM4q50TuORvLpGTgT8MCsqC4pqxC4pqrNeh0nbThLs'
client = Client(api_key, api_secret)

tickers: list[dict[str, str]] = client.get_all_tickers()
candle_data: list[list[Union[float, int]]]
selected_tickers: list[str] = [
    x['symbol'] for x in tickers if 'USDT' == x['symbol'][-4:] and not (x['symbol'][-8:-4] == 'DOWN' or x['symbol'][-6:-4] == 'UP')
]
"""
with open("C:\\Users\\andri\\Documents\\Programming\\Algorithmic trading\\broadened_feature_timeseries.csv",
          'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "pair symbol", "open timestamp", "open", "high", "low", "close", "volume", "close_timestamp",
        "quote_volume", "trade_quantity", "taker_buy_base", "taker_buy_quote"
    ])
"""
for index_, instrument_ in enumerate(selected_tickers[140:155]):
    cycle_start = time.time()
    try:
        candle_data = client.get_historical_klines(instrument_, Client.KLINE_INTERVAL_1HOUR, "1 Jan 2020", "25 Sep 2022")
    except Exception as e:
        timestamp = client._get_earliest_valid_timestamp(instrument_, '1h')
        print(f"{e}{new_line}at {instrument_}")
        candle_data = client.get_historical_klines(instrument_, Client.KLINE_INTERVAL_1HOUR, timestamp, "18 Dec 2021")

    with open("C:\\Users\\andri\\Documents\\Programming\\Algorithmic trading\\broadened_feature_timeseries2.csv", 'a+', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        for candle_ in candle_data:
            writer.writerow([f'{instrument_}', f'{time.ctime(int(candle_[0]) / 1000)}', *[candle_[1:-1]]])

    print(f'{instrument_}, {index_ + 1}/{len(selected_tickers)} done, cycle duration: {time.time() - cycle_start}')

# /\: avg time spent is too long, mul by #cycles -> async for << async func
# TD: shift header by adding <ccy_nm>