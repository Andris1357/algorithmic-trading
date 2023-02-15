# WAS WRITTEN IN PYTHON 3.10
from binance.client import Client
import time
from datetime import datetime, timedelta
import pyodbc
from binance.enums import *
import pandas as pd
import numpy as np
from functools import reduce
from asgiref.sync import sync_to_async
import asyncio
from binance.exceptions import *
from typing import Union

def binance_connect(network) -> Client:
    if network == 'main':
        api_key = '???' # sensitive value masked
        api_secret = '???' # sensitive value masked
        return Client(api_key, api_secret)
    elif network == 'test':
        api_key = 'GzO8JbXPI43LNa4mF9C4xJsB01V3ZXBhu2sq8VwljTNfHa6PMwLlDAkqmAJtjTS5'  # change to current testnet credentials if deleted (each month, so these are almost certainly deprecated here)
        api_secret = 'lzRGSXyGEHVgyfuGBzhuJGlsRj4XT8CF7xbogVpHV1cQPXPdHRw1S1kvs1tH38XE'
        _client = Client(api_key, api_secret)
        _client.API_URL = 'https://testnet.binance.vision/api'
        return _client
client = binance_connect('main')

def getTemplate(file_str):
    file = pd.read_excel(file_str)
    return np.array([[[float(file.iat[row_i + block_i * (file.shape[1]), col_i + 1]) for col_i in
                       range(file.shape[1] - 1)] for row_i in range(file.shape[1] - 1)] for block_i in
                     range(file.shape[1] - 1)])

ccy_names = pd.read_excel("C:\\Users\\andri\\Documents\\Programming\\alg_templates\\ccy_list.xlsx")
heatmap_long_struct = [getTemplate(f"C:\\Users\\andri\\Documents\\Programming\\alg_templates\\{ccy_name}_heatmap_long.xlsx") for ccy_name in ccy_names.iloc[:, 1]]
heatmap_short_struct = [getTemplate(f"C:\\Users\\andri\\Documents\\Programming\\alg_templates\\{ccy_name}_heatmap_short.xlsx") for ccy_name in ccy_names.iloc[:, 1]]
extremes_nnls = [[[np.min(heatmap_long_struct[ccy]), np.max(heatmap_long_struct[ccy])], [np.min(heatmap_short_struct[ccy]), np.max(heatmap_short_struct[ccy])]] for ccy in range(ccy_names.shape[0])]
scales_long_struct = [{'chg': (scales_src := pd.read_excel(f"C:\\Users\\andri\\Documents\\Programming\\alg_templates\\{ccy_nm}scales_long.xlsx")).iloc[:, 1],
          'vol': scales_src.iloc[:, 2], 'pr': scales_src.iloc[:, 3]} for ccy_nm in ccy_names.iloc[:, 1]]
scales_short_struct = [{'chg': (scales_src := pd.read_excel(
    f"C:\\Users\\andri\\Documents\\Programming\\alg_templates\\alg_templates\\{ccy_nm}scales_short.xlsx")).iloc[:, 1],
                 'vol': scales_src.iloc[:, 2], 'pr': scales_src.iloc[:, 3]} for ccy_nm in ccy_names.iloc[:, 1]]
tresholds = pd.read_excel("C:\\Users\\andri\\Documents\\Programming\\alg_templates\\tresholds.xlsx") # WILL BE CALCULATED AS AN OPTIMAL PERCENTAGE LEVEL WHERE IF WE SEPARATED THE ASCENDING ORDERED LIST OF HEATMAP VALUES, WE WOULD INCLUDE ENOUGH CASES SO WE DO NOT TRADE INFREQUENTLY, BUT NOT TOO MUCH SO IT WILL PICK PROFITABLE ENOUGH COORDINATES
tresholds = [[tresholds.iloc[side + 1, ccy] for ccy in range(tresholds.shape[0])] for side in range(2)]

newl = '\n'
log_obj_ls = []
freq = int(15)
ccy_names_ls = ccy_names.iloc[:, 1]
init_data = list([[float(), int(), float()] for x in range(len(ccy_names.iloc[:, 1]))])
pr_hist = list([list() for _ in range(len(ccy_names.iloc[:, 1]))])
in_trade = list(['' for _ in range(len(ccy_names.iloc[:, 1]))])
stoploss_win, stoploss_lose, bid_floor = 0.15, 0.06, 12

db_name = '???'
driver = '{SQL Server Native Client 11.0}'
server = '???'
user = '???'
pwd = ''
conn_str = f'DRIVER={driver};SERVER={server};DATABASE={db_name};Trusted_connection=yes'
connection = pyodbc.connect(conn_str)
db_handle = connection.cursor()
backtest = True


class DebugMsg:
    session = datetime.now()

    def __init__(self, id_, cycle_, nature: str, type_, content, timestamp_):
        self.id = id_
        self.cycle = cycle_
        self.nature = nature  # 'ERROR'|'INFO'
        self.type = type_
        self.content = content
        self.timest = timestamp_

class Balance:
    test = backtest
    busd = 1000.
    usdt = 0.
    ccy_balances = [0. for _ in ccy_names.iloc[:, 1]]
    executing_client = None

    @classmethod
    def updateBalance(cls) -> None:
        
        if not cls.test:
            cls.busd = client.get_asset_balance(asset="BUSD")['free']
            cls.usdt = client.get_asset_balance(asset="USDT")['free']
        else:
            ...
    @classmethod
    def updateTestBalances(cls, _ccy: int, _amt: float) -> None:
        if cls.test:
            cls.ccy_balances[_ccy] = _amt
    @classmethod
    def addExecClient(cls):
        return binance_connect('test')


def f_ccy_pr(_pair: str) -> float:
    return client.get_symbol_ticker(symbol=_pair + "USDT")['price']

def f_calc_bid(bal: float, _in_trade: list[bool], _asset_info_arr: list[list][str|int, int|float, int|float, int]) -> float|int:
    if bal < 10 * 0.5 * len(list(filter(lambda y: y == '' and _asset_info_arr[y][3] not in [1, 2], _in_trade))):  #change in_trade to input th receives ones in trade, correct at all func calls
        return 10
    else:
        if bal * 0.024 > 12:
            return bal * 0.024
        elif bal < 10:
            return 0
        elif 10 <= bal < bid_floor:
            return bal
        else:
            return bid_floor

def getBuyingPower(_balance = Balance) -> float:
    if not _balance.test:
        _balance.updateBalance()
    return _balance.busd + _balance.usdt

def f_ccy_bal(_ccy: str, _balance = Balance) -> float:
    return client.get_asset_balance(_ccy)['free'] if not Balance.test else Balance.ccy_balances[ccy_names.iloc[:, 1].index(_ccy)] #list(filter(lambda y: _ccy == , range(0, len(to_watch))))[0]

def f_wealth(_last_prices: Union[list[float], int], _balance = Balance) -> float:
    outstanding = sum([f_ccy_bal(ccy_names.iloc[x, 1]) * _last_prices[x] for x in range(ccy_names.shape[0])]) if not _balance.test else [
        Balance.ccy_balances[x2] * f_ccy_pr(ccy_names.iloc[x2, 1]) for x2 in range(len(Balance.ccy_balances))]
    return getBuyingPower(_balance) + outstanding

def gen_id_tnx(session):
    x = 0
    while True:
        yield f'{session}/{x}'
        x += 1
f_gen_id_tnx = gen_id_tnx(DebugMsg.session)

def gen_id_log():
    z = 0
    while True:
        yield z
        z += 1
f_gen_id_log = gen_id_log()

def f_get_constraints(_asset: str) -> list[str|int, int|float, int|float, int]:
    only_usdt = int()

    try:
        symbol_info = client.get_symbol_info(symbol=_asset)["filters"]
    except TypeError as error:
        print(error)
        try:
            symbol_info = client.get_symbol_info(symbol=_asset + "USDT")
            
            only_usdt = 1
            Balance.ccy_balances[ccy_names.iloc[:, 1].index(_asset)] = ccy_names.iloc[list(ccy_names.iloc[:, 1]).index(_asset)] + 'USDT'
            log_obj_ls.append(
                DebugMsg(next(f_gen_id_log), 0, 'I', 'type', f"{_asset} only has USDT pair", datetime.now()))
        except TypeError as error:
            log_obj_ls.append(DebugMsg(next(f_gen_id_log), 0, 'E', 'type', f"{_asset} does not exist; message: {error}",
                                       datetime.now()))
            only_usdt = 2
            return ['invalid', -1, -1, only_usdt]

    step_val = list(filter(lambda x: x["filterType"] == "LOT_SIZE", symbol_info))[0]["stepSize"]
    step_val_ls = list(step_val)
    dot_index = step_val_ls.index('.')

    if any(x != '0' for x in step_val_ls[dot_index + 1::]):
        trim_zeros = 0
        while step_val_ls[trim_zeros - 1] == '0':
            trim_zeros -= 1
        symbol_decimals = len(step_val_ls[dot_index + 1:trim_zeros:])
        assert (step_val_ls[dot_index + 1:trim_zeros:][-1] == '1'), f"step value (float) is not an exponent of 10 at {_asset}"
    else:
        symbol_decimals = 0
        assert (step_val_ls[0] == '1'), f"step value (integer) is not an exponent of 10 at {_asset}"

    if dot_index != 1:
        symbol_decimals = -1 * len(step_val_ls[:dot_index:]) - 1 
    notional_min = float(list(filter(lambda x: x["filterType"] == "MIN_NOTIONAL", symbol_info))[0]["minNotional"])
    currency_min = float(list(filter(lambda x: x["filterType"] == "LOT_SIZE", symbol_info))[0]["minQty"])

    return [symbol_decimals, notional_min, currency_min, only_usdt]

def f_fin_amt(asset: str, pr: float, ccy_amt: float|str, decimals: int, _bid: float, usdt: int, side: str, _cycle: int, _balance = Balance) -> float:
    fin_amt = -1.

    if side == 'ask':
        if _balance.test:
            ccy_bal_temp = _balance.ccy_balances[list(ccy_names.iloc[:, 1]).index(asset)]
        else:
            ccy_bal_temp = client.get_asset_balance(asset)['free']
        if ccy_amt == 'liquidate':
            fin_amt = (lambda x: x if x <= ccy_bal_temp else x - 10 ** (-1 * decimals))(round(ccy_bal_temp, decimals))
        else:
            if ccy_bal_temp * pr / 3 >= 10:  # if we are already working with bigger volume, increase value of withdrawal from 10 to a dynamic % value based on value of buying power
                fin_amt = (lambda x: x if x <= ccy_bal_temp else x + 10 ** (-1 * decimals))(round(ccy_bal_temp / 3, decimals))
            elif ccy_bal_temp * pr > 10 + 10 * (1 + stoploss_win):
                fin_amt = (lambda x: x if x <= ccy_bal_temp else x + 10 ** (-1 * decimals))(round(10 / pr, decimals))  # sell 10 dollars worth
            else:
                fin_amt = (lambda x: x if x <= ccy_bal_temp else x - 10 ** (-1 * decimals))(round(ccy_bal_temp, decimals))
        while any(fin_amt < x for x in [asset_info_arr[ccy_names_ls.index(asset)][1] * pr, asset_info_arr[ccy_names_ls.index(asset)][2]]):  # will refer to USDT data in terms of respective assets
            fin_amt += 10 ** (-1 * decimals)

    elif side == 'bid': # do not always exchange back USDT (mounting fees) -> use USDT to buy assets when there is enough amount of it; calculate balance with USDT taken into account
        if usdt == 1:
            _balance.updateBalance()
            temp_usdt_in = _balance.usdt
            fin_amt = round((ccy_amt * pr - temp_usdt_in) * pr, decimals)
        else:
            fin_amt = round(ccy_amt, decimals)
        while any(fin_amt < x for x in [asset_info_arr[ccy_names_ls.index(asset)][1] * pr, asset_info_arr[ccy_names_ls.index(asset)][2]]):  # will refer to USDT data in terms of respective assets
            fin_amt += 10 ** (-1 * decimals)
        if _bid * 0.9 > fin_amt or fin_amt > _bid * 1.5:
            log_obj_ls.append(DebugMsg(next(f_gen_id_log), _cycle, 'I', 'finance',
                                       f"final price is off by {fin_amt - _bid} at {asset}", datetime.now()))

    assert (side in ['ask', 'bid']), f"invalid side specified at {asset}, {side}"
    return fin_amt

def f_order(side: bool, _pair: str, _amt: float, _cycle: int, _balance = Balance) -> int|None|list[float, float, float]:  # if there are more fills because of rapidly changing prices, take an average of transaction price & delivered amount of currency
    if Balance.test:
        if side:
            _balance.usdt, _balance.busd = (lambda y: [_balance.usdt + _amt * client.get_symbol_ticker(symbol=_pair), _balance.busd] if y == 'USDT' else [_balance.usdt, _balance.busd + _amt * client.get_symbol_ticker(symbol=ccy)])(ccy[-4::])
            Balance.updateTestBalances(list(ccy_names.iloc[:, 1]).index(_pair[:-4]), _amt)
        else:
            _balance.usdt, _balance.busd = (lambda y: [_balance.usdt - _amt * client.get_symbol_ticker(symbol=_pair), _balance.usdt] if y == 'USDT' else [_balance.usdt, _balance.busd - _amt * client.get_symbol_ticker(symbol=ccy)])(ccy[-4::])
            Balance.updateTestBalances(list(ccy_names.iloc[:, 1]).index(_pair[:-4]), _amt)
    else:
        getBuyingPower()

    with (Balance.executing_client if Balance.test else client) as context_client:
        if side:  # handle expired order registries, try again if intended to sell, also no liquidity makes closing urgent -> do not buy any assets if there is no liquidity, market price may be unpredictable
            order = context_client.order_market_buy(symbol=_pair, quantity=_amt)
            if order['status'] == 'EXPIRED':
                log_obj_ls.append(DebugMsg(next(f_gen_id_log), _cycle, 'E', 'finance', f"no liquidity at {_pair}", datetime.now()))
                return -1
            elif order['status'] != 'EXPIRED' and order['fills'][0] == {}:
                log_obj_ls.append(DebugMsg(next(f_gen_id_log), _cycle, 'E', 'type', f"unknown error at {_pair} with status {order['status']}", datetime.now()))
                return -1
            else:
                ...
        else:
            order = context_client.order_market_sell(symbol=_pair, quantity=_amt)
            if order['status'] == 'EXPIRED':
                i = 0
                while order['status'] == 'EXPIRED' and i < 5:
                    order = context_client.order_market_sell(symbol=_pair, quantity=_amt)
                    i += 1
                if order['status'] == 'EXPIRED' and i >= 5:
                    log_obj_ls.append(DebugMsg(next(f_gen_id_log), _cycle, 'E', 'finance', f"failed to sell due to insufficient liquidity at {_pair}, {_amt}", datetime.now()))
                    return -1
                else:
                    ...

    if len(order['fills']) == 1:
        fin_pr = order['fills'][0]['price'] 
        fees = order['fills'][0]['commission'] * (lambda y: client.get_symbol_ticker(symbol=y + 'USDT') if y != 'USDT' else 1)(order['commissionAsset'])['price']
        quan = order['fills'][0]['qty']
    else:
        quan = sum([x['qty'] for x in order['fills']])
        fin_pr = sum([x['price'] * x['qty'] / quan for x in order['fills']])
        fees = sum([x['commission'] * (lambda y: client.get_symbol_ticker(symbol=y + 'USDT') if y != 'USDT' else 1)(x['commissionAsset']) for x in order['fills']])['price']

    return [fin_pr, quan, fees]

@sync_to_async
def callSqlAsync(_error_i: DebugMsg, _db_handle = db_handle):
    db_handle.execute(f"INSERT INTO $table_errors ($err_id, $session, $time, $type, $text, $cycle) VALUES"
                      f"{newl}({_error_i.id}, CONVERT(DATETIME, SUBSTRING('{_error_i.session}', 0, 24), 121), "
                       f"CONVERT(DATETIME, SUBSTRING('{_error_i.timest}', 0, 24), 121), '{_error_i.nature}', '{_error_i.type}', {_error_i.cycle})")
    db_handle.commit()
    return "success"

async def f_log_messages(error_objs: list[DebugMsg]) -> None:
    if len(error_objs) != 0:  # if due to any error the cycle fails, try blocks within the code will not let error messages revert either error objects or them being appended to the error list
        futures = []
        for error_i in error_objs:
            futures.append(asyncio.ensure_future(callSqlAsync(error_i)))
        await asyncio.gather(*futures, return_exceptions=True)


cycle = int(0)
asset_info_arr = list([f_get_constraints(x) for x in ccy_names.iloc[:, 1]])
print("Getting constraints done")
balance_log_mod = 1
start_balance = getBuyingPower()
start_time = datetime.now()
session_on = bool(1)
finish = bool(0)
emergency_finish = bool(0)
i_out = None
purge_mkt_data = 200  # so we can only look back this many candles for each coin, but this becomes a 100 limit after a purge
# if there are open tnx-s fr the previous run, identify them and store their parameters to init_data variable; get tnx rows where there wasnt a closing type tnx => get $id, $ticker, $time, $side, $price, $amt | id (-> in_trade), set cycle_passed to their datetime subtracted from curr datetime or 10min from the one with the earliest datetime
db_handle.execute(
    f"SELECT $price, $amt, $ticker, $time FROM $table_tnx AS t_tnx"
    f"INNER JOIN (SELECT MIN($time) AS first_open, $id FROM $table_tnx WHERE $side = 'open' GROUP BY $id) AS open_tnx ON open_tnx.first_open = t_tnx AND t_tnx.id = earliest_tnx.id")
init_rows = db_handle.fetchall()
if init_rows != []:
    for row in init_rows:
        init_data[ccy_names_ls.index(list(filter(lambda y: row[2][0:-4] in y, ccy_names_ls))[0])] = [row[0], int((lambda z: z.days * 240 + round(
            z.seconds / 600, 0))(datetime.now() - row[3])), row[1]]

while True:
    try:
        if cycle % 100 == 99:
            db_handle = connection.cursor()  # probably there is some timeout on the db, it is wise to refresh the conn every once in a while
        take_profit, confidence, pr_i = False, float(), float()
        start = datetime.now()
        t_cnt = time.perf_counter()
        next_cycle = start + timedelta(minutes=freq)
        if f_wealth([x[-1] for x in pr_hist] if not Balance.test else -1) < start_balance * 0.9:  # later can bring it down to 0.8
            emergency_finish = 1  # break alg and setup limit orders for trades in profit, close ones losing
        try:  # breaks automatically if wealth fell below a certain % over the session compared to the cycle starting amount
            for ccy in range(ccy_names.shape[0]):
                try:
                    i_out, error_i_ls, last_candles = ccy, [], None  # either this w ß.join | class w list compr sql insert
                    if not Balance.test:
                        Balance.updateBalance()
                    # if not enough busd, switch buys to usdt
                    # EARLY ORDER CLOSE MANAGEMENT, FREEING UP CAPITAL -> flag certain orders "to close" if a certain ratio of buying power against assets outstanding (open transactions) si reached
                    if getBuyingPower() < 10 * 0.5 * len(list(filter(lambda y: y == '' and asset_info_arr[y][3] != 2, in_trade))):
                        take_profit = True

                    if asset_info_arr[ccy][3] in [0, 1]:  # checks if ccy exists
                        if asset_info_arr[ccy][3] == 0:
                            pr_i = client.get_symbol_ticker(symbol=ccy_names.iloc[:, 1][ccy])["price"]
                            last_candles = client.get_historical_klines(ccy_names.iloc[:, 1][ccy] + "BUSD", Client.KLINE_INTERVAL_15MINUTE, "6 hours 15 minutes ago UTC")[-24:] # D: make dates dyn <= (curr cycle time).{day, mo, y}
                        elif asset_info_arr[ccy][3] == 1:
                            pr_i = client.get_symbol_ticker(symbol=ccy_names.iloc[:, 1][ccy])["price"]
                            last_candles = client.get_historical_klines(ccy_names.iloc[:, 1][ccy] + "USDT", Client.KLINE_INTERVAL_15MINUTE, "6 hours 15 minutes ago UTC")[-24:]

                        aggr_candle_data = {'close': last_candles[-1][4], 'volume': sum([last_candles[x][5] for x in range(24)]), 'chg': last_candles[-1][4] - last_candles[0][1]}
                                             
                        within_range, par_matches = [[scale_i[ccy][x][0] <= aggr_candle_data[x] <= scale_i[ccy][x][-1] for x in aggr_candle_data] for scale_i in [scales_long_struct, scales_short_struct]], [[None, None] for _ in aggr_candle_data] #  approximates current candle features to be fit into feature value distribution
                        lower_neighbor_long, higher_neighbor_long, lower_neighbor_short, higher_neighbor_short = [[int() for _ in aggr_candle_data] for _ in range(4)]
                        findParMatches = lambda y: [lower_neighbor_long[y] if (par_val_i := aggr_candle_data[list(aggr_candle_data.keys())[y]]) - lower_neighbor_long[y] < higher_neighbor_long[y] - par_val_i else higher_neighbor_long, 
                                    lower_neighbor_short[y] if par_val_i - lower_neighbor_short[y] < higher_neighbor_short[y] - par_val_i else higher_neighbor_short[y]]
                        getNeighbors, val_split_ext, index_long, index_short = lambda y: [[lower_neighbor_long[y], higher_neighbor_long[y]], [lower_neighbor_short[y], higher_neighbor_short[y]]], 1, [], []

                        if not all(within_range): # I: IF VAL IS OUTSIDE SCALE LABELS, TREAT IT AS THE CLOSER EXTREME
                            for par_label_i in aggr_candle_data.keys():
                                if not within_range[list(aggr_candle_data.keys()).index(par_label_i)][0]: # LONG
                                    if all(aggr_candle_data[par_label_i] > x for x in scales_long_struct[ccy][par_label_i]):
                                        index_long.append(-1)
                                    else: # val is below lower bound
                                        index_long.append(0)
                                else:
                                    lower_neighbor_long[list(aggr_candle_data.keys()).index(par_label_i)] = list(
                                        filter(lambda y: aggr_candle_data[par_label_i] >= y,
                                               scales_long_struct[ccy][par_label_i]))[-1]
                                    higher_neighbor_long[list(aggr_candle_data.keys()).index(par_label_i)] = list(
                                        filter(lambda y: aggr_candle_data[par_label_i] <= y,
                                               scales_long_struct[ccy][par_label_i]))[0]

                                if not within_range[list(aggr_candle_data.keys()).index(par_label_i)][1]: # SHORT
                                    if all(aggr_candle_data[par_label_i] > x for x in scales_short_struct[ccy][par_label_i]):
                                        index_short.append(-1)
                                    else: # val is below lower bound
                                        index_short.append(0)
                                else:
                                    lower_neighbor_short[list(aggr_candle_data.keys()).index(par_label_i)] = list(
                                        filter(lambda y: aggr_candle_data[par_label_i] >= y,
                                               scales_short_struct[ccy][par_label_i]))[-1]
                                    higher_neighbor_short[list(aggr_candle_data.keys()).index(par_label_i)] = list(
                                        filter(lambda y: aggr_candle_data[par_label_i] <= y,
                                               scales_short_struct[ccy][par_label_i]))[0]
                            confidence = [heatmap_long_struct[ccy][tuple(index_long)], heatmap_short_struct[ccy][tuple(index_short)]]

                        else: # STANDARD CASE, VAL IS WITHIN THE SCALE OF PARAM VALUE DISTRIBUTION
                            for par_label_i in aggr_candle_data.keys():
                                lower_neighbor_long[list(aggr_candle_data.keys()).index(par_label_i)] = list(filter(lambda y: aggr_candle_data[par_label_i] >= y, scales_long_struct[ccy][par_label_i]))[-1]
                                higher_neighbor_long[list(aggr_candle_data.keys()).index(par_label_i)] = list(filter(lambda y: aggr_candle_data[par_label_i] <= y, scales_long_struct[ccy][par_label_i]))[0]
                                lower_neighbor_short[list(aggr_candle_data.keys()).index(par_label_i)] = list(filter(lambda y: aggr_candle_data[par_label_i] >= y, scales_short_struct[ccy][par_label_i]))[-1]
                                higher_neighbor_short[list(aggr_candle_data.keys()).index(par_label_i)] = list(filter(lambda y: aggr_candle_data[par_label_i] <= y, scales_short_struct[ccy][par_label_i]))[0]
                            if not any(is_equal := [[(par_val_i := aggr_candle_data[list(aggr_candle_data.keys())[x]]) - lower_neighbor_long[x] == higher_neighbor_long[x] - par_val_i for x in range(len(aggr_candle_data))],
                                        [(par_val_i := aggr_candle_data[list(aggr_candle_data.keys())[x]]) - lower_neighbor_short[x] == higher_neighbor_short[x] - par_val_i for x in range(len(aggr_candle_data))]]):
                                par_matches = [findParMatches(x) for x in range(len(aggr_candle_data))]
                                index_long = [scales_long_struct[ccy][list(scales_long_struct[ccy].keys())[x]].index(par_matches[x][0]) for x in range(len(par_matches))] # which is the index of the value of iterator match_i => pass as *{args}
                                index_short = [scales_short_struct[ccy][list(scales_short_struct[ccy].keys())[x]].index(par_matches[x][1]) for x in range(len(par_matches))]
                                confidence = [heatmap_long_struct[ccy][tuple(index_long)], heatmap_short_struct[ccy][tuple(index_short)]]

                            else:
                                for par_i in range(len(aggr_candle_data)):
                                    if not any(is_equal[x][par_i] for x in range(2)):
                                        par_matches[par_i] = findParMatches(par_i)
                                    elif all(is_equal[x][par_i] for x in range(2)):
                                        par_matches[par_i] = getNeighbors(par_i)
                                    else:
                                        which_eq = list(filter(lambda y: not is_equal[y][par_i], range(2)))[0]
                                        par_matches[par_i][which_eq] = findParMatches(par_i)[which_eq]
                                        par_matches[par_i][-~-which_eq] = getNeighbors(par_i)[-~-which_eq] # invert 0 to 1 and vica versa using negate(bitwise not(negate()))
                                val_split_ext = reduce(lambda y, z: y * z, map(len, par_matches))

                                index_long = [scales_long_struct[ccy][list(scales_long_struct[ccy].keys())[x]].index(par_matches[x][0])
                                     if not par_matches[x] is list else [scales_long_struct[ccy][list(scales_long_struct[ccy].keys())[x]].index(par_matches[x][0][0]),
                                                                         scales_long_struct[ccy][list(scales_long_struct[ccy].keys())[x]].index(par_matches[x][0][1])] for x in range(len(par_matches))]
                                index_short = [scales_short_struct[ccy][list(scales_short_struct[ccy].keys())[x]].index(
                                    par_matches[x][0])
                                              if not par_matches[x] is list else [
                                    scales_short_struct[ccy][list(scales_short_struct[ccy].keys())[x]].index(
                                        par_matches[x][0][0]),
                                    scales_short_struct[ccy][list(scales_short_struct[ccy].keys())[x]].index(
                                        par_matches[x][0][1])] for x in range(len(par_matches))]

                                confidence = [sum(heatmap_long_struct[ccy][tuple(index_long)]) / val_split_ext,
                                              sum(heatmap_short_struct[ccy][tuple(index_short)]) / val_split_ext] # NEVER CHOOSE BOTH, [OCO(`>`)]!

                        # approximate to between 2 scale distribution values, choose either one that is closer; get indices of all matching scale vals; evaluated both short and long side & decides which is better
                        
                        pr_hist[ccy].append(pr_i)
                        bid_fiat = f_calc_bid(getBuyingPower(), in_trade, asset_info_arr)
                        bid, side_fork = f_fin_amt(ccy_names.iloc[ccy, 1], pr_i, bid_fiat / pr_i, asset_info_arr[ccy][0], bid_fiat, asset_info_arr[ccy][3], 'bid', cycle), -1
                        if any(confidence[x] >= tresholds[x][ccy] for x in range(2)) and in_trade[ccy] == '' and finish == 0 and emergency_finish == 0:
                            
                            if all(confidence[x] >= tresholds[x][ccy] for x in range(2)):
                                side_fork = confidence.index(max(confidence))
                            else:
                                side_fork = list(filter(lambda y: confidence[y] >= tresholds[y][ccy], range(2)))[0]
                            if confidence[0] >= tresholds[0][ccy] and side_fork == 0:
                                for _ in [1]:
                                    if asset_info_arr[ccy][3] == 1:  # check if there is enough USDT, exchange if not
                                        if bid * pr_i > Balance.usdt * 1.05:
                                            f_order(False, "BUSDUSDT", round((lambda: bid * pr_i * 1.01 - Balance.usdt if bid * pr_i * 1.01 - Balance.usdt > 10 else 10)(), 4), cycle)
                                    elif asset_info_arr[ccy][3] == 0:
                                        if bid * pr_i > Balance.busd and Balance.usdt >= bid:
                                            f_order(True, "BUSDUSDT", round((lambda: bid * pr_i * 1.01 - Balance.busd if bid * pr_i * 1.01 - Balance.busd > 10 else 10)(), 0), cycle)  # round according to decimals value (4,0) -> less if overflow
                                    else:
                                        continue
                                    # exchange to USDT if there isnt enough stored

                                    try:
                                        order_details = f_order(True, (lambda x: ccy_names.iloc[ccy, 1] if x == 0 else (ccy_names_ls[ccy][:-4:] + "USDT"))(asset_info_arr[ccy][3]), bid, cycle)  # if usdt=1, change symbol
                                        if order_details != -1:  # if there is an error, f_order itself will notify of potential cause, no exception handling needed here
                                            in_trade[ccy] = next(f_gen_id_tnx)
                                            init_data[ccy][0] = order_details[0]  # for measuring individual order profitability; [1] can be used for making decisions when freeing up room for new trades & also when running lower on cash
                                            init_data[ccy][1] = cycle  # init_data stores initial stats of tnx
                                            init_data[ccy][2] = order_details[1]  # this stored ccy amount, ~[0] stores pr
                                            db_handle.execute(
                                                f"INSERT INTO $table_tnx ($id, $ticker, $time, $side, $price, $amt, $slippage, $comm) VALUES('{in_trade[ccy]}', '{ccy_names_ls[ccy]}', "
                                                f"CONVERT(DATETIME, SUBSTRING('{datetime.now()}', 0, 24), 121), 'open', '{order_details[0]}', '{order_details[1]}', '{1 - pr_i / init_data[ccy][0]}'), {order_details[2]}")
                                            # wil yield negative value for slippage which turned out to be favorable (gained on the transaction); $fraction will be 0 until any amount is sold from the stack
                                            connection.commit()
                                    except Exception as ex:
                                        log_obj_ls.append(DebugMsg(next(f_gen_id_log), cycle, 'E', 'order failed',
                                                                   f"order failed with {bid}, {ccy_names_ls[ccy]}, price {client.get_symbol_ticker(symbol=ccy_names_ls[ccy])['price']}, "
                                                                   f"when balance was {f_ccy_bal((lambda: 'USDT' if asset_info_arr[ccy][3] == 1 else 'BUSD')())} due to: {newl}{ex}",
                                                                   datetime.now()))
                            if confidence[1] >= tresholds[1][ccy] and side_fork == 1:
                                # TO BE IMPLEMENTED; CALCULATING THE BUYING POWER ALLOCATION FOR SHORTING WILL BE ALONG A DIFFERENT LOGIC DUE TO THE MINIMUM 3X LEVERAGE ON BINANCE; DUMMY ORDER ˇˇ
                                margin_sell_order = client.create_margin_order(symbol='BNBBTC', side=SIDE_SELL, type=ORDER_TYPE_MARKET, timeInForce=TIME_IN_FORCE_GTC, quantity=100, price='0.00001', isIsolated='TRUE')

                        elif emergency_finish == 1:
                            for liquidate_i in [in_trade.index(x) for x in list(filter(lambda y: y != '', in_trade))]:
                                try:
                                    order_details = f_order(False,
                                                            (lambda x: ccy_names_ls[liquidate_i] if x == 0 else (ccy_names_ls[ccy][:-4:] + "USDT"))(asset_info_arr[liquidate_i][3]),
                                                            f_fin_amt(ccy_names_ls[liquidate_i],
                                                                      f_ccy_pr(ccy_names.iloc[liquidate_i, 1]), 'liquidate',
                                                                      asset_info_arr[liquidate_i][0], -1,
                                                                      asset_info_arr[liquidate_i][3], 'ask', cycle), cycle)
                                    db_handle.execute(
                                        
                                        f"INSERT INTO $table_tnx ($id, $ticker, $time, $side, $price, $amt, $slippage, $comm, $r_rat, $r_scal) "
                                        f"VALUES('{in_trade[ccy]}', '{ccy_names_ls[ccy]}', {datetime.now()}, 'close', {order_details[0]}, '{order_details[1]}', {1 - order_details[0] / pr_i}, {order_details[2]}, "
                                        f"{order_details[0] / init_data[ccy][0] - 1}, {init_data[ccy][2] * init_data[ccy][0] - order_details[1] * order_details[0]}")
                                except Exception as ex:
                                    log_obj_ls.append(DebugMsg(next(f_gen_id_log), cycle, 'E', 'order failed',
                                                               f"order failed with {bid}, {ccy_names_ls[liquidate_i]}, price {client.get_symbol_ticker(symbol=ccy_names_ls[liquidate_i])['price']}, "
                                                               f"when balance was {f_ccy_bal((lambda: 'USDT' if asset_info_arr[liquidate_i][3] == 1 else 'BUSD')())} by error {newl}{ex}", datetime.now()))
                                    print(f"could not liquidate {ccy_names_ls[liquidate_i]}")
                                connection.commit()

                        elif in_trade[ccy] != '': # currency is in trade, may be closed
                            # knock out if open transactions' profit per cycle drops below a certain point
                            ask = f_fin_amt(ccy_names.iloc[:, 1][ccy], pr_i, client.get_asset_balance(ccy_names_ls[ccy][0:-4])['free'], asset_info_arr[ccy][0], -1, asset_info_arr[ccy][3], 'ask', cycle)
                            db_handle.execute(f"SELECT $cycle FROM $table_tnx WHERE $id = '{in_trade[ccy]}' AND $side = 'open'")
                            cycle_temp = db_handle.fetchall()[0][0]

                            if cycle - cycle_temp >= 144 and pr_i / pr_hist[ccy][-13] < 1.04:  # and their value did not increase too steeply in the last few hours
                                try:
                                    order_details = f_order(False, (
                                        lambda x: ccy_names.iloc[ccy, 1] if x == 0 else (ccy_names.iloc[ccy, 1] + "USDT"))(asset_info_arr[ccy][3]), ask, cycle)
                                    db_handle.execute(
                                        f"INSERT INTO $table_tnx ($id, $ticker, $time, $side, $price, $amt, $slippage, $comm, $r_rat, $r_scal) "
                                        f"VALUES('{in_trade[ccy]}', '{ccy_names.iloc[:, 1][ccy]}', {datetime.now()}, 'close', {order_details[0]}, '{order_details[1]}', {1 - order_details[0] / pr_i}, {order_details[2]}, "
                                        f"{order_details[0] / init_data[ccy][0] - 1}, {init_data[ccy][2] * init_data[ccy][0] - order_details[1] * order_details[0]}")
                                    connection.commit()
                                    in_trade[ccy] = ''
                                except:
                                    log_obj_ls.append(DebugMsg(next(f_gen_id_log), cycle, 'E', 'order failed',
                                                               f"order failed with {ask}, {ccy_names.iloc[ccy, 1]}, price {client.get_symbol_ticker(symbol=ccy_names.iloc[ccy, 1])['price']}, "
                                                               f"when balance was {f_ccy_bal((lambda: 'USDT' if asset_info_arr[ccy][3] == 1 else 'BUSD')())}",
                                                               datetime.now()))

                            else:
                                if take_profit:  # sell from every position where you have been over a certain profitability/hr over the course of the trade
                                    for to_profit in list(filter(lambda y: ((f_ccy_pr((lambda z: z[0:-4] + 'USDT' if asset_info_arr[y][3] == 1 else z)(y)) /
                                                                             init_data[y][0] - 1) / (cycle - init_data[y][1]) / 6 > 0.02 and cycle - init_data[y][1] > 6 * 4)
                                                                           or f_ccy_pr(y) / init_data[y][0] - 1 > 0.5, ccy_names.iloc[:, 1])):  # TD:
                                        f_order(False, to_profit, (lambda w: (lambda: w / 3 if w / 3 >= 10 else 10)() if 10 + 10 * (1 + stoploss_win) > w else w)(
                                            f_ccy_bal(ccy_names_ls[ccy_names_ls.index(to_profit)][0:-4])), cycle)
                                    for to_profit_margin in # SHORT SELLING, TO BE FINISHED
                                        result = client.cancel_margin_order(symbol="MATICBUSD", orderId='?')
                                elif init_data[ccy][0] * (1 - stoploss_lose) >= pr_i or pr_i <= max(
                                        pr_hist[ccy][init_data[ccy][1]:cycle + 1:]) * (1 - stoploss_win) or (pr_i >= init_data[ccy][0] * 1.5 and client.get_asset_balance(
                                    ccy_names.iloc[ccy, 1][0:-4])['free'] * pr_i > 25):
                                    
                                    # CLOSE ORDER IF A CERTAIN NUMBER OF CYCLES HAVE PASSED; EMERGENCY EXIT = close all, the losing ones may lose more & the winning ones might give back profit
                                    try:
                                        order_details = f_order(False, (lambda x: ccy_names.iloc[ccy, 1] + "BUSD" if x == 0 else (ccy_names.iloc[ccy, 1] + "USDT"))(asset_info_arr[ccy][3]), ask, cycle)
                                        if order_details != -1:
                                            db_handle.execute(f"SELECT CASE WHEN EXISTS (SELECT * FROM $table_tnx WHERE $id = '{in_trade[ccy]}' AND $side = 'close') THEN 1 ELSE 0")
                                            assert (len(db_handle.fetchall()) > 1), "db_handle.fetchall yields more rows than 1 in case of 'EXISTS' expr"
                                            close_tnx_query = db_handle.fetchall()[0][0]
                                            fractd_tnx = (lambda: -1 if not f_ccy_bal(ccy_names.iloc[ccy, 1][0:-4]) >= 10 ** (asset_info_arr[ccy][0] * -1) else in_trade[2] / order_details[1])()

                                            if close_tnx_query == 0 and fractd_tnx == -1:  # means that there were no previous tnx-s & this tnx closed in one increment -> in_trade reset, $fract = null
                                                db_handle.execute(
                                                    f"INSERT INTO $table_tnx ($id, $ticker, $time, $side, $price, $amt, $slippage, $comm, $r_rat, $r_scal) "
                                                    f"VALUES('{in_trade[ccy]}', '{ccy_names_ls[ccy]}', {datetime.now()}, 'close', {order_details[0]}, '{order_details[1]}', {1 - order_details[0] / pr_i}, {order_details[2]}, "
                                                    f"{order_details[0] / init_data[ccy][0] - 1}, {init_data[ccy][2] * init_data[ccy][0] - order_details[1] * order_details[0]}")
                                                connection.commit()
                                                in_trade[ccy] = ''
                                            elif close_tnx_query == 1 and fractd_tnx == -1:  # there were prev tnx-s but this was the last one -> in_trade reset, $fract => calc
                                                db_handle.execute(
                                                    f"INSERT INTO $table_tnx ($id, $ticker, $time, $side, $price, $amt, $slippage, $comm, $r_rat, $r_scal, $fract)"
                                                    f"VALUES ('{in_trade[ccy]}', '{ccy_names.iloc[ccy, 1]}', {datetime.now()}, 'close', {order_details[0]}, '{order_details[1]}', {1 - order_details[0] / pr_i}, {order_details[2]}, "
                                                    f"{order_details[0] / init_data[ccy][0] - 1}, {order_details[1] * order_details[0] - init_data[ccy][2] * init_data[ccy][0] * fractd_tnx}, {fractd_tnx})")
                                                connection.commit()
                                                in_trade[ccy] = ''
                                            elif close_tnx_query == 1 and fractd_tnx != -1:  # tnx has already been fractionally closed \ there is still left of the ccy after this tnx -> in_trade stays, else as 1 above
                                                db_handle.execute(
                                                    f"INSERT INTO $table_tnx ($id, $ticker, $time, $side, $price, $amt, $slippage, $comm, $r_rat, $r_scal, $fract)"
                                                    f"VALUES ('{in_trade[ccy]}', '{ccy_names.iloc[ccy, 1]}', {datetime.now()}, 'close', {order_details[0]}, '{order_details[1]}', {1 - order_details[0] / pr_i}, {order_details[2]}, "
                                                    f"{order_details[0] / init_data[ccy][0] - 1}, {order_details[1] * order_details[0] - init_data[ccy][2] * init_data[ccy][0] * fractd_tnx}, {fractd_tnx})")
                                                connection.commit()
                                            else:
                                                log_obj_ls.append(DebugMsg(next(f_gen_id_log), cycle, 'E', 'logic',
                                                                           f"invalid combination of close_tnx_query = {close_tnx_query} & fractd_tnx = {fractd_tnx} at {ccy_names.iloc[ccy, 1]}, tnx {in_trade[ccy]}", datetime.now()))
                                    except Exception as exc:
                                        log_obj_ls.append(DebugMsg(next(f_gen_id_log), cycle, 'E', 'order failed',
                                                                   f"order failed by {exc} with {ask}, {ccy_names.iloc[ccy, 1]}, price {client.get_symbol_ticker(symbol=ccy_names.iloc[ccy, 1] + 'USDT')['price']}, "
                                                                   f"when balance was {f_ccy_bal((lambda: 'USDT' if asset_info_arr[ccy][3] == 1 else 'BUSD')())}", datetime.now()))

                    elif asset_info_arr[ccy][3] == 2:
                        log_obj_ls.append(DebugMsg(next(f_gen_id_log), cycle, 'I', 'finance', f"asset {ccy_names.iloc[ccy, 1]} skipped", datetime.now()))
                    else:
                        log_obj_ls.append(DebugMsg(next(f_gen_id_log), cycle, 'E', 'user', f"wrong value of {asset_info_arr[ccy][3]} at {ccy_names_ls[ccy]}", datetime.now()))
                except Exception as exc:
                    log_obj_ls.append(DebugMsg(next(f_gen_id_log), cycle, 'E', 'cycle skipped', f"cycle {cycle} failed due to to '{exc}'", datetime.now()))
                finally:
                    if emergency_finish == 1:
                        break

            db_handle.execute(f"INSERT INTO $table_finance ($type, $cycle, $time, $wealth, $cash, $usdt, $open_tnx) "
                              f"VALUES ('cycle_level', {cycle}, CONVERT(DATETIME, SUBSTRING('{datetime.now()}', 0, 24), 121), {f_wealth([x[-1] for x in pr_hist] if not Balance.test else -1)}, {getBuyingPower()}, {f_ccy_bal('USDT')}, {len(list(filter(lambda x: x != '', in_trade)))})")
            if balance_log_mod % 6 == 0:
                print(f"BUSD/full: {client.get_asset_balance('BUSD')['free']}/{getBuyingPower()}; wealth:{f_wealth([x[-1] for x in pr_hist] if not Balance.test else -1)}")
            balance_log_mod += 1
            time.sleep((next_cycle - datetime.now()).total_seconds())
            cycle += 1

        except KeyboardInterrupt:  # keyboardinterrupt will get caught in inner ßtry block -> it doesnt count as that, only gradually exits; second keyboardinterrupt will archive debug messages then break execution
            print(f"exit|gradual at {datetime.now()}")
            db_handle.execute(f"INSERT INTO $table_finance ($type, $cycle, $time, $wealth, $session_r, $open_tnx"
                              f"VALUES('session_level', {cycle}, CONVERT(DATETIME, SUBSTRING('{datetime.now()}', 0, 24), 121), {getBuyingPower()}")
            print("finish process started")
            finish = 1
    except KeyboardInterrupt:
        print(f"exit|instant at {datetime.now()}")
        
        f_log_messages(log_obj_ls)
        alert_str = ''
        for trade_i in list(filter(lambda x: x != '', in_trade)):
            pair = ccy_names_ls[ccy_names_ls.index(trade_i)]
            alert_str += '\n' + f"{pair}, P&L: {init_data[i_out][0] / float(client.get_symbol_ticker(symbol=pair)['price']) - 1}, amt: {f_ccy_bal(pair)}"
        print(f"still open: {alert_str}")
        break  # second kbinterrupt will finish loop without running last scripts

    finally: # logging of errors happens in all circumstances
        event_loop = asyncio.new_event_loop()
        try: # gather coroutines
            event_loop.run_until_complete(f_log_messages(log_obj_ls))
        finally:
            event_loop.close()
        print(f"Algorithm finished running at {datetime.now()}")

    if emergency_finish == 1 or (finish == 1 and len(list(filter(lambda x: x != '', in_trade))) == 0):
        break
    if len(pr_hist[0]) > purge_mkt_data:
        for i2 in pr_hist:
            del i2[0:100]
