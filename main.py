#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate.io(芝麻开门) 市场数据量化脚本 —— 定时任务版
-------------------------------------------------
此脚本每 3 秒自动执行一次，获取距离当前 3 秒以前、最近 30 分钟的 K 线，
并分析 EMA、RSI、VWAP、放量/缩量，给出做多/做空/观望结论。

运行：
    python gateio_3s_kline_scheduler.py

说明：
- 使用 Gate 公共行情接口(无需 API Key)，不会下单；仅供学习研究。
- 使用标准库实现，无需第三方库安装。
"""

import json
import math
import time
import ssl
import urllib.parse
import urllib.request
from collections import deque
from typing import Dict, List, Tuple
from datetime import datetime
import sched

GATE_BASE = "https://api.gateio.ws/api/v4"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "gate-quant-scheduler/1.0"
}

scheduler = sched.scheduler(time.time, time.sleep)

# ------------------------- HTTP -------------------------

def http_get(path: str, params: Dict[str, str]) -> List:
    qs = urllib.parse.urlencode(params)
    url = f"{GATE_BASE}{path}?{qs}"
    req = urllib.request.Request(url, headers=HEADERS, method="GET")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
        data = resp.read()
        return json.loads(data.decode("utf-8"))

# ------------------------- 指标 -------------------------

def ema(values: List[float], span: int) -> List[float]:
    if not values:
        return []
    k = 2 / (span + 1)
    out = []
    ema_val = values[0]
    for v in values:
        ema_val = v * k + ema_val * (1 - k)
        out.append(ema_val)
    return out

def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < 2:
        return [50.0] * len(values)
    rsis = [50.0]
    avg_gain = 0.0
    avg_loss = 0.0
    gains, losses = [], []
    for i in range(1, min(period + 1, len(values))):
        chg = values[i] - values[i - 1]
        gains.append(max(chg, 0.0))
        losses.append(max(-chg, 0.0))
    avg_gain = sum(gains) / max(1, len(gains))
    avg_loss = sum(losses) / max(1, len(losses))
    while len(rsis) < period:
        rs = (avg_gain / (avg_loss + 1e-12)) if avg_loss > 0 else 9999.0
        rsis.append(100 - 100 / (1 + rs))
    for i in range(period + 1, len(values)):
        chg = values[i] - values[i - 1]
        gain, loss = max(chg, 0.0), max(-chg, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = (avg_gain / (avg_loss + 1e-12)) if avg_loss > 0 else 9999.0
        rsis.append(100 - 100 / (1 + rs))
    return rsis[:len(values)]

def rolling_vwap(close: List[float], volume: List[float], window: int) -> List[float]:
    vwap = []
    sum_pv, sum_v = 0.0, 0.0
    dq_pv, dq_v = deque(), deque()
    for i in range(len(close)):
        pv = close[i] * volume[i]
        dq_pv.append(pv)
        dq_v.append(volume[i])
        sum_pv += pv
        sum_v += volume[i]
        if len(dq_pv) > window:
            sum_pv -= dq_pv.popleft()
            sum_v -= dq_v.popleft()
        vwap.append(sum_pv / sum_v if sum_v > 0 else float("nan"))
    return vwap

def volume_state(volume: List[float], window: int = 20, up_thr: float = 1.2, down_thr: float = 0.8) -> Tuple[str, float]:
    if len(volume) < window:
        return ("样本不足", float("nan"))
    base = sum(volume[-window:]) / window
    latest = volume[-1]
    ratio = latest / (base + 1e-12)
    if ratio >= up_thr:
        return ("放量", ratio)
    elif ratio <= down_thr:
        return ("缩量", ratio)
    else:
        return ("正常", ratio)

def wick_body(candle: Dict[str, float]) -> Dict[str, float]:
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    return {"upper": h - max(o, c), "lower": min(o, c) - l, "body": abs(c - o)}

# ------------------------- 数据抓取 -------------------------

def fetch_candles(symbol: str, interval: str, from_ts: int, to_ts: int, limit: int = 1000) -> List[Dict]:
    path = "/spot/candlesticks"
    params = {"currency_pair": symbol, "interval": interval, "from": from_ts, "to": to_ts, "limit": limit}
    raw = http_get(path, params)
    rows = []
    for item in raw:
        ts, qv, close, high, low, open_, bv, finished = item
        rows.append({
            "ts": int(float(ts)),
            "time": datetime.utcfromtimestamp(int(float(ts))).strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(open_), "high": float(high), "low": float(low), "close": float(close),
            "base_vol": float(bv), "quote_vol": float(qv), "finished": str(finished).lower() == "true"
        })
    rows.sort(key=lambda x: x["ts"])
    return rows

# ------------------------- 交易逻辑 -------------------------

def decide_action(candles: List[Dict], ema_fast: int, ema_slow: int, breakout: int, vol_window: int) -> Dict:
    if len(candles) < max(ema_fast, ema_slow, breakout, vol_window) + 1:
        return {"action": "观望", "reason": "样本不足", "extras": {}}
    close, high, low, vol = [c["close"] for c in candles], [c["high"] for c in candles], [c["low"] for c in candles], [c["base_vol"] for c in candles]
    ema_f, ema_s, rsi14, vwap_w = ema(close, ema_fast), ema(close, ema_slow), rsi(close, 14), rolling_vwap(close, vol, vol_window)
    last = candles[-1]
    trend_up, trend_dn = ema_f[-1] > ema_s[-1], ema_f[-1] < ema_s[-1]
    recent_high, recent_low = max(high[-breakout:]), min(low[-breakout:])
    vol_stat, v_ratio = volume_state(vol, vol_window)
    w = wick_body(last)
    long_setup = trend_up and last["close"] > recent_high and vol_stat == "放量" and not (w["upper"] > 2 * w["body"])
    short_setup = trend_dn and last["close"] < recent_low and vol_stat == "放量" and not (w["lower"] > 2 * w["body"])
    action = "做多" if long_setup else ("做空" if short_setup else "观望")
    return {"action": action, "reason": f"趋势{'上' if trend_up else ('下' if trend_dn else '震荡')} + 突破/跌破 + {vol_stat}",
            "extras": {"price": last["close"], "recent_high": recent_high, "recent_low": recent_low, "ema_fast": ema_f[-1], "ema_slow": ema_s[-1], "rsi14": rsi14[-1], "vwap": vwap_w[-1], "volume_state": vol_stat, "volume_ratio": v_ratio, "upper_wick": w["upper"], "lower_wick": w["lower"], "body": w["body"]}}

# ------------------------- 定时任务 -------------------------

def task(symbol: str, interval: str, window_min: int, ema_fast: int, ema_slow: int, breakout: int, vol_window: int, poll: int):
    to_ts, from_ts = int(time.time()) - 3, int(time.time()) - 3 - window_min * 60
    candles = fetch_candles(symbol, interval, from_ts, to_ts)
    if not candles:
        print(f"[{datetime.now()}] 无数据，稍后再试…")
    else:
        res = decide_action(candles, ema_fast, ema_slow, breakout, vol_window)
        ex = res["extras"]
        print(f"[{datetime.now()}] {symbol} => {res['action']} | {res['reason']} | 价格={ex['price']:.4f}")
    scheduler.enter(poll, 1, task, (symbol, interval, window_min, ema_fast, ema_slow, breakout, vol_window, poll))

if __name__ == "__main__":
    symbol, interval, poll, window_min = "ETH_USDT", "1m", 3, 30
    ema_fast, ema_slow, breakout, vol_window = 9, 21, 10, 20
    print(f"启动定时任务，每 {poll} 秒执行一次")
    scheduler.enter(0, 1, task, (symbol, interval, window_min, ema_fast, ema_slow, breakout, vol_window, poll))
    scheduler.run()
