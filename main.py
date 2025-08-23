#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate.io(芝麻开门) 3秒轮询 + 回测 一体脚本（零依赖）v2
=================================================
为何有这个版本：
- 你反馈“从 4825 跌到 4807 还提示观望”。原因多半是：没有达到放量阈值/没有击穿最近低点/出现极端影线，旧版策略偏保守。
- 本版新增 **动量破位模型**（Momentum）来捕捉快速下跌/上冲，不强制要放量；并可与原 **突破+量能模型**（Breakout）并用。
- 同时内置 **2分钟持有回测**，评估胜率和期望。

特性：
1) 每 3 秒抓一次，默认使用【已收盘的最后一根】K 线计算信号（避免半根K线抖动）。
2) 两套信号：
   - Breakout：趋势(EMA9/21) + 近N根突破/跌破 + (可选)放量确认
   - Momentum：近期收益Z分数/ATR冲量 + EMA同向，抓“滑坡式”快速行情（即 4825→4807 这类）
3) 2分钟固定持有（可配），支持 **实盘提示** 与 **历史回测** 两种模式。
4) 纯标准库，无需安装三方包。

使用示例：
- 实盘（10秒K，加速捕捉 1~2 分钟机会）：
    python gateio_quant_v2.py live --symbol ETH_USDT --interval 10s --poll 3 --window 30 \
        --mode both --hold 120 --vol_thr 1.05 --z_thr 1.8 --atr_mult 0.6

- 回测（近 2 天，10秒K，持有 120s）：
    python gateio_quant_v2.py backtest --symbol ETH_USDT --interval 10s --days 2 \
        --mode both --hold 120 --vol_thr 1.05 --z_thr 1.8 --atr_mult 0.6 --export trades.csv

参数要点：
- --mode: breakout/momentum/both  三选一（默认 both）
- --hold: 持有秒数（默认 120 = 2分钟）
- --vol_thr: 放量阈值倍数（默认 1.2；若设 0 或 1 表示不要求放量）
- --z_thr: 动量Z分数阈值（越小越激进；常见 1.6~2.2）
- --atr_mult: ATR 冲量倍数阈值（越小越激进；常见 0.5~1.0）

风险提示：仅供学习研究，不构成投资建议。请始终自设风控与止损。
"""

import argparse
import json
import math
import time
import ssl
import urllib.parse
import urllib.request
from collections import deque
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

GATE_BASE = "https://api.gateio.ws/api/v4"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "gate-quant-v2/1.0"
}

# ------------------------- HTTP -------------------------

def http_get(path: str, params: Dict[str, str]) -> List:
    qs = urllib.parse.urlencode(params)
    url = f"{GATE_BASE}{path}?{qs}"
    req = urllib.request.Request(url, headers=HEADERS, method="GET")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
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
    avg_gain, avg_loss = 0.0, 0.0
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


def atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
    if len(close) == 0:
        return []
    trs = [0.0]
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        trs.append(tr)
    # 简化版EMA平滑
    return ema(trs, period)


def zscore(series: List[float], lookback: int) -> List[float]:
    out = []
    for i in range(len(series)):
        if i + 1 < lookback:
            out.append(0.0)
            continue
        window = series[i+1-lookback:i+1]
        mean = sum(window) / lookback
        var = sum((x - mean) ** 2 for x in window) / max(1, lookback - 1)
        std = math.sqrt(max(var, 1e-12))
        out.append((series[i] - mean) / std if std > 0 else 0.0)
    return out

# ------------------------- 数据抓取 -------------------------

def interval_to_seconds(interval: str) -> int:
    m = interval.strip().lower()
    if m.endswith("s"): return int(m[:-1])
    if m.endswith("m"): return int(m[:-1]) * 60
    if m.endswith("h"): return int(m[:-1]) * 3600
    if m.endswith("d"): return int(m[:-1]) * 86400
    raise ValueError(f"不支持的 interval: {interval}")


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
    rows.sort(key=lambda x: x["ts"])  # 升序
    return rows


def fetch_candles_window(symbol: str, interval: str, from_ts: int, to_ts: int) -> List[Dict]:
    out: List[Dict] = []
    sec_per_bar = interval_to_seconds(interval)
    # 按 1000 根一段抓取
    cur_from = from_ts
    while cur_from < to_ts:
        cur_to = min(to_ts, cur_from + sec_per_bar * 1000)
        part = fetch_candles(symbol, interval, cur_from, cur_to, 1000)
        if part:
            out.extend(part)
            cur_from = part[-1]["ts"] + sec_per_bar
        else:
            break
    return out

# ------------------------- 策略逻辑 -------------------------

# ======== 1分钟方向预测（零依赖简化版） ========
# 思路：对最近一小段价格做线性回归取斜率（Slope），结合最近收益的Z分数（MomentumZ）、
# EMA快慢线的距离（EMAGap）以及RSI斜率（RSI Drift），做一个加权投票，输出未来 ~60 秒
# 上/下/震荡 的概率与方向。这里是“**短时趋势判断**”，不是价格点位预测。
#
# 可调参数：
#   horizon_secs: 预测未来的秒数（默认 60s）
#   lookback_mult: 回看窗口大约是 horizon 的倍数（默认 5 倍）
#   up_thr / dn_thr: 概率阈值（默认 0.55 / 0.45）

class ForecastParams:
    def __init__(self, horizon_secs:int = 60, lookback_mult:float = 5.0,
                 up_thr:float = 0.55, dn_thr:float = 0.45):
        self.horizon_secs = horizon_secs
        self.lookback_mult = lookback_mult
        self.up_thr = up_thr
        self.dn_thr = dn_thr


def _ols_slope_tstat(y: List[float]) -> Tuple[float, float]:
    """对等间隔时间序列 y 做一元线性回归，返回 (斜率, t统计量)。"""
    n = len(y)
    if n < 3:
        return 0.0, 0.0
    x_mean = (n-1)/2.0
    y_mean = sum(y)/n
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = i - x_mean
        num += dx * (y[i] - y_mean)
        den += dx * dx
    slope = num / (den + 1e-12)
    # 残差方差
    ss = 0.0
    for i in range(n):
        y_hat = y_mean + slope * (i - x_mean)
        ss += (y[i] - y_hat) ** 2
    sigma2 = ss / max(1, n - 2)
    se = math.sqrt(sigma2 / (den + 1e-12))
    t = slope / (se + 1e-12)
    return slope, t


def _zscore_last(series: List[float], lb: int) -> float:
    if len(series) < lb:
        return 0.0
    window = series[-lb:]
    mean = sum(window)/lb
    var = sum((x-mean)**2 for x in window)/max(1, lb-1)
    std = math.sqrt(max(var,1e-12))
    return (series[-1]-mean)/(std+1e-12)


def forecast_direction_1m(candles: List[Dict], sec_per_bar: int, fp: ForecastParams) -> Dict:
    if not candles:
        return {"dir": "flat", "prob_up": 0.5, "prob_dn": 0.5, "explain": "无数据"}
    # 只用已收盘K线
    if not candles[-1].get("finished", True):
        candles = candles[:-1]
    close = [c["close"] for c in candles]
    rsi14 = rsi(close, 14)
    # 回看窗口按 horizon 的倍数取
    horizon_bars = max(1, int(round(fp.horizon_secs / max(1, sec_per_bar))))
    lookback = max(5, int(round(horizon_bars * fp.lookback_mult)))
    tail = close[-lookback:]
    # 1) 斜率 + t 值
    slope, tstat = _ols_slope_tstat(tail)
    # 2) 近K收益 z 分数
    rets = [0.0]
    for i in range(1, len(close)):
        rets.append((close[i]-close[i-1])/(close[i-1]+1e-12))
    z_mo = _zscore_last(rets, min(len(rets), max(5, lookback//2)))
    # 3) EMA Gap
    ema_f = ema(close, 9)
    ema_s = ema(close, 21)
    emagap = (ema_f[-1] - ema_s[-1])/(close[-1]+1e-12)
    # 4) RSI Drift（最近若干根的斜率）
    rsi_tail = rsi14[-min(lookback, len(rsi14)) : ]
    _, rsi_t = _ols_slope_tstat(rsi_tail)

    # 将各分量映射到 0-1 概率（sigmoid），进行加权
    def sigmoid(x):
        return 1.0/(1.0+math.exp(-x))
    p_slope = sigmoid(tstat)
    p_momo  = sigmoid(2.5*z_mo)
    p_gap   = sigmoid(30*emagap)
    p_rsi   = sigmoid(rsi_t/2.0)

    # 权重：斜率 0.4，动量 0.3，EMA 0.2，RSI 0.1
    w1,w2,w3,w4 = 0.4,0.3,0.2,0.1
    prob_up = max(0.0, min(1.0, w1*p_slope + w2*p_momo + w3*p_gap + w4*p_rsi))
    prob_dn = 1.0 - prob_up

    if prob_up >= fp.up_thr:
        d = "up"
    elif prob_dn >= (1.0 - fp.dn_thr):
        d = "down"
    else:
        d = "flat"

    return {
        "dir": d,
        "prob_up": prob_up,
        "prob_dn": prob_dn,
        "meta": {
            "horizon_secs": fp.horizon_secs,
            "lookback": lookback,
            "tstat": tstat,
            "z_momentum": z_mo,
            "ema_gap": emagap,
            "rsi_t": rsi_t
        },
        "explain": f"Slope-t={tstat:.2f}, Z={z_mo:.2f}, EMAGap={emagap:.4f}, RSI-t={rsi_t:.2f}"
    }


class Params:
    def __init__(self, ema_fast=9, ema_slow=21, breakout=10, vol_window=20, vol_thr=1.2,
                 z_lookback=12, z_thr=1.8, atr_period=14, atr_mult=0.6, hold_secs=120,
                 mode="both"):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.breakout = breakout
        self.vol_window = vol_window
        self.vol_thr = vol_thr
        self.z_lookback = z_lookback
        self.z_thr = z_thr
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.hold_secs = hold_secs
        self.mode = mode  # breakout/momentum/both


def volume_ratio(vol: List[float], window: int) -> float:
    if len(vol) < window:
        return float("nan")
    base = sum(vol[-window:]) / window
    return vol[-1] / (base + 1e-12)


def decide_action(candles: List[Dict], p: Params, sec_per_bar: int) -> Dict:
    # 仅使用最后一根“已收盘”的K线来生成信号
    if not candles:
        return {"action": "观望", "reason": "无数据", "extras": {}}
    if not candles[-1]["finished"]:
        candles = candles[:-1]
    if len(candles) < max(p.ema_fast, p.ema_slow, p.breakout, p.vol_window, p.atr_period, p.z_lookback) + 1:
        return {"action": "观望", "reason": "样本不足", "extras": {}}

    close = [c["close"] for c in candles]
    high = [c["high"] for c in candles]
    low = [c["low"] for c in candles]
    vol = [c["base_vol"] for c in candles]

    ema_f = ema(close, p.ema_fast)
    ema_s = ema(close, p.ema_slow)
    rs = rsi(close, 14)
    atr_arr = atr(high, low, close, p.atr_period)

    # 计算收益与Z分数（动量窗口 z_lookback，10sK 配 12 ≈ 2分钟）
    rets = [0.0]
    for i in range(1, len(close)):
        rets.append((close[i] - close[i-1]) / (close[i-1] + 1e-12))
    z = zscore(rets, p.z_lookback)

    last = candles[-1]
    price = last["close"]
    trend_up, trend_dn = ema_f[-1] > ema_s[-1], ema_f[-1] < ema_s[-1]
    recent_high = max(high[-p.breakout:])
    recent_low = min(low[-p.breakout:])
    vratio = volume_ratio(vol, p.vol_window)

    # ---------- Breakout 信号 ----------
    allow_novol = (p.vol_thr <= 1.0 or math.isnan(vratio))
    bo_long = trend_up and price > recent_high and (allow_novol or vratio >= p.vol_thr)
    bo_short = trend_dn and price < recent_low and (allow_novol or vratio >= p.vol_thr)

    # ---------- Momentum 信号 ----------
    impulse_dn = (price <= close[-3] - p.atr_mult * (atr_arr[-1] or 0)) and trend_dn
    impulse_up = (price >= close[-3] + p.atr_mult * (atr_arr[-1] or 0)) and trend_up
    z_dn = z[-1] <= -p.z_thr and trend_dn
    z_up = z[-1] >= p.z_thr and trend_up
    mo_short = impulse_dn or z_dn
    mo_long = impulse_up or z_up

    use_bo = (p.mode in ("breakout", "both"))
    use_mo = (p.mode in ("momentum", "both"))

    reasons = []
    action = "观望"
    if use_mo and mo_short:
        action = "做空"
        reasons.append("Momentum: 快速下跌/负Z分数")
    if use_bo and bo_short and action == "观望":
        action = "做空"; reasons.append("Breakout: 跌破近低点" + ("+放量" if not allow_novol else ""))
    if use_mo and mo_long and action == "观望":
        action = "做多"; reasons.append("Momentum: 快速上冲/正Z分数")
    if use_bo and bo_long and action == "观望":
        action = "做多"; reasons.append("Breakout: 突破近高点" + ("+放量" if not allow_novol else ""))

    # 极端影线过滤（仅在与方向相反的影线极长时谨慎）
    def wick_body(c):
        o,h,l,c2 = c["open"],c["high"],c["low"],c["close"]
        return {"upper": h - max(o,c2), "lower": min(o,c2) - l, "body": abs(c2 - o)}
    w = wick_body(last)
    if action == "做多" and w["upper"] > 2*w["body"]:
        reasons.append("上影异常→观望")
        action = "观望"
    if action == "做空" and w["lower"] > 2*w["body"]:
        reasons.append("下影异常→观望")
        action = "观望"

    hold_bars = max(1, int(math.ceil(p.hold_secs / sec_per_bar)))

    return {
        "action": action,
        "reason": "; ".join(reasons) if reasons else ("无触发" if action=="观望" else ""),
        "extras": {
            "price": price,
            "ema_fast": ema_f[-1],
            "ema_slow": ema_s[-1],
            "rsi14": rs[-1],
            "atr": atr_arr[-1],
            "vratio": vratio,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "zscore_last": z[-1],
            "hold_bars": hold_bars
        }
    }

# ------------------------- 回测 -------------------------

def backtest(candles: List[Dict], p: Params, sec_per_bar: int, fee_bp: float = 5.0, slip_bp: float = 5.0,
             export_path: Optional[str] = None) -> Dict:
    """固定持有 p.hold_secs；入场价=下一根开盘价；出场价=持有N根后的开盘价。
    手续费与滑点采用基点(bps)；默认单边各 5 bps（0.05%）。
    """
    if not candles:
        return {"trades": [], "summary": {}}

    # 仅用已收盘K线
    cands = [c for c in candles if c.get("finished", True)]
    sec_per_bar = sec_per_bar

    trades = []
    i = max(p.ema_fast, p.ema_slow, p.breakout, p.vol_window, p.atr_period, p.z_lookback)
    while i < len(cands) - 2:  # 确保有下一根开盘 & 持有期
        window = cands[:i+1]
        sig = decide_action(window, p, sec_per_bar)
        action = sig["action"]
        if action in ("做多", "做空"):
            entry_bar = i + 1
            hold_bars = sig["extras"]["hold_bars"]
            exit_bar = min(len(cands)-1, entry_bar + hold_bars)
            entry_price = cands[entry_bar]["open"]
            exit_price = cands[exit_bar]["open"]
            direction = 1 if action == "做多" else -1
            # 费用与滑点（进+出双边各一次）
            cost_rate = (fee_bp + slip_bp) / 10000.0 * 2
            raw_ret = direction * (exit_price / entry_price - 1.0)
            net_ret = raw_ret - cost_rate
            trades.append({
                "entry_time": cands[entry_bar]["time"],
                "exit_time": cands[exit_bar]["time"],
                "side": action,
                "entry": entry_price,
                "exit": exit_price,
                "raw_ret": raw_ret,
                "net_ret": net_ret,
                "reason": sig["reason"],
            })
            # 跳到平仓后下一根，避免重叠持仓
            i = exit_bar + 1
        else:
            i += 1

    # 汇总
    rets = [t["net_ret"] for t in trades]
    win = sum(1 for r in rets if r > 0)
    loss = sum(1 for r in rets if r <= 0)
    cum = 1.0
    equity = []
    for r in rets:
        cum *= (1 + r)
        equity.append(cum)
    mdd = 0.0
    peak = 1.0
    for e in equity:
        peak = max(peak, e)
        mdd = max(mdd, (peak - e) / peak)
    avg = sum(rets)/len(rets) if rets else 0.0
    var = sum((x-avg)**2 for x in rets)/max(1,len(rets)-1)
    std = math.sqrt(max(var,1e-12))
    sharpe = (avg/std*math.sqrt(252*24*60*60/p.hold_secs)) if std>0 else 0.0  # 粗略折算

    summary = {
        "trades": len(trades),
        "win_rate": win/max(1,win+loss),
        "avg_net_ret": avg,
        "cum_return": cum-1.0,
        "max_drawdown": mdd,
        "sharpe_like": sharpe,
    }

    if export_path:
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                f.write("entry_time,exit_time,side,entry,exit,raw_ret,net_ret,reason\n")
                for t in trades:
                    f.write(f"{t['entry_time']},{t['exit_time']},{t['side']},{t['entry']},{t['exit']},{t['raw_ret']},{t['net_ret']},\"{t['reason']}\"\n")
        except Exception as e:
            print(f"导出失败: {e}")

    return {"trades": trades, "summary": summary}

# ------------------------- 实盘循环 -------------------------

def live_loop(symbol: str, interval: str, poll: int, window_min: int, p: Params):
    sec_per_bar = interval_to_seconds(interval)
    fp = ForecastParams(horizon_secs=60, lookback_mult=5.0, up_thr=0.55, dn_thr=0.45)
    print(f"[LIVE] {symbol} interval={interval} poll={poll}s window={window_min}m mode={p.mode} hold={p.hold_secs}s")
    while True:
        try:
            to_ts = int(time.time()) - 3
            from_ts = to_ts - window_min * 60
            candles = fetch_candles_window(symbol, interval, from_ts, to_ts)
            if not candles:
                print(f"[{datetime.now()}] 无数据")
            else:
                sig = decide_action(candles, p, sec_per_bar)
                ex = sig["extras"]
                fc = forecast_direction_1m(candles, sec_per_bar, fp)
                print(
                    f"[{datetime.now()}] {symbol} => {sig['action']} | {sig['reason']} | "
                    f"P={ex['price']:.2f} EMA{p.ema_fast}/{p.ema_slow}={ex['ema_fast']:.2f}/{ex['ema_slow']:.2f} "
                    f"RSI14={ex['rsi14']:.1f} ATR={ex['atr']:.2f} Z={ex['zscore_last']:.2f} "
                    f"VR={ex['vratio'] if ex['vratio']==ex['vratio'] else float('nan'):.2f} | "
                    f"[1m预测] {fc['dir']}  up={fc['prob_up']:.2%} dn={fc['prob_dn']:.2%}  ({fc['explain']})"
                )
        except Exception as e:
            print(f"异常: {type(e).__name__}: {e}")
        finally:
            time.sleep(max(1, poll))

    sec_per_bar = interval_to_seconds(interval)
    print(f"[LIVE] {symbol} interval={interval} poll={poll}s window={window_min}m mode={p.mode} hold={p.hold_secs}s")
    while True:
        try:
            to_ts = int(time.time()) - 3
            from_ts = to_ts - window_min * 60
            candles = fetch_candles_window(symbol, interval, from_ts, to_ts)
            if not candles:
                print(f"[{datetime.now()}] 无数据")
            else:
                sig = decide_action(candles, p, sec_per_bar)
                ex = sig["extras"]
                print(
                    f"[{datetime.now()}] {symbol} => {sig['action']} | {sig['reason']} | "
                    f"P={ex['price']:.2f} EMA{p.ema_fast}/{p.ema_slow}={ex['ema_fast']:.2f}/{ex['ema_slow']:.2f} "
                    f"RSI14={ex['rsi14']:.1f} ATR={ex['atr']:.2f} Z={ex['zscore_last']:.2f} "
                    f"VR={ex['vratio'] if ex['vratio']==ex['vratio'] else float('nan'):.2f} HoldBars={ex['hold_bars']}"
                )
        except Exception as e:
            print(f"异常: {type(e).__name__}: {e}")
        finally:
            time.sleep(max(1, poll))

# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Gate.io 3秒轮询 + 回测 一体脚本（零依赖）v2")
    sub = p.add_subparsers(dest="cmd", required=True)

    # live
    pl = sub.add_parser("live", help="实盘信号")
    pl.add_argument("--symbol", default="ETH_USDT")
    pl.add_argument("--interval", default="10s")
    pl.add_argument("--poll", type=int, default=3)
    pl.add_argument("--window", type=int, default=30)
    pl.add_argument("--mode", choices=["breakout","momentum","both"], default="both")
    pl.add_argument("--hold", type=int, default=120)
    pl.add_argument("--ema_fast", type=int, default=9)
    pl.add_argument("--ema_slow", type=int, default=21)
    pl.add_argument("--breakout", type=int, default=10)
    pl.add_argument("--vol_window", type=int, default=20)
    pl.add_argument("--vol_thr", type=float, default=1.2)
    pl.add_argument("--z_lookback", type=int, default=12)
    pl.add_argument("--z_thr", type=float, default=1.8)
    pl.add_argument("--atr_period", type=int, default=14)
    pl.add_argument("--atr_mult", type=float, default=0.6)

    # backtest
    pb = sub.add_parser("backtest", help="历史回测")
    pb.add_argument("--symbol", default="ETH_USDT")
    pb.add_argument("--interval", default="10s")
    pb.add_argument("--days", type=int, default=2)
    pb.add_argument("--mode", choices=["breakout","momentum","both"], default="both")
    pb.add_argument("--hold", type=int, default=120)
    pb.add_argument("--ema_fast", type=int, default=9)
    pb.add_argument("--ema_slow", type=int, default=21)
    pb.add_argument("--breakout", type=int, default=10)
    pb.add_argument("--vol_window", type=int, default=20)
    pb.add_argument("--vol_thr", type=float, default=1.05)
    pb.add_argument("--z_lookback", type=int, default=12)
    pb.add_argument("--z_thr", type=float, default=1.8)
    pb.add_argument("--atr_period", type=int, default=14)
    pb.add_argument("--atr_mult", type=float, default=0.6)
    pb.add_argument("--fee_bp", type=float, default=5.0)
    pb.add_argument("--slip_bp", type=float, default=5.0)
    pb.add_argument("--export", default="")

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "live":
        p = Params(
            ema_fast=args.ema_fast, ema_slow=args.ema_slow, breakout=args.breakout,
            vol_window=args.vol_window, vol_thr=args.vol_thr, z_lookback=args.z_lookback,
            z_thr=args.z_thr, atr_period=args.atr_period, atr_mult=args.atr_mult,
            hold_secs=args.hold, mode=args.mode
        )
        live_loop(args.symbol, args.interval, args.poll, args.window, p)
    else:
        # 回测时间窗
        to_ts = int(time.time()) - 3
        from_ts = to_ts - max(1, args.days) * 86400
        sec_per_bar = interval_to_seconds(args.interval)
        candles = fetch_candles_window(args.symbol, args.interval, from_ts, to_ts)
        p = Params(
            ema_fast=args.ema_fast, ema_slow=args.ema_slow, breakout=args.breakout,
            vol_window=args.vol_window, vol_thr=args.vol_thr, z_lookback=args.z_lookback,
            z_thr=args.z_thr, atr_period=args.atr_period, atr_mult=args.atr_mult,
            hold_secs=args.hold, mode=args.mode
        )
        res = backtest(candles, p, sec_per_bar, fee_bp=args.fee_bp, slip_bp=args.slip_bp,
                       export_path=(args.export if args.export else None))
        s = res["summary"]
        print("--- 回测结果 ---")
        print(f"交易笔数: {s.get('trades',0)}  胜率: {s.get('win_rate',0):.2%}")
        print(f"平均净收益/笔: {s.get('avg_net_ret',0):.4%}  累计收益: {s.get('cum_return',0):.2%}")
        print(f"最大回撤: {s.get('max_drawdown',0):.2%}  Sharpe-like: {s.get('sharpe_like',0):.2f}")
        if args.export:
            print(f"明细已导出: {args.export}")


if __name__ == "__main__":
    main()
