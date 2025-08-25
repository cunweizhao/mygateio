#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate.io 一体化：历史回测 + 指标分析(BOLL/MACD/量能/支撑压力) + 实盘每秒轮询 + 3秒线性预测
纯标准库，无第三方依赖。Python 3.8+
用法示例：
  - 历史回测（近2天，ETH_USDT，10sK）:
      python gateio_quant_one.py backtest --symbol ETH_USDT --interval 10s --days 2
  - 实盘轮询（每秒输出一次信号，并给出3秒预测）:
      python gateio_quant_one.py live --symbol ETH_USDT --interval 10s
环境提示：若你的机器证书存在问题，可临时设置：
  PowerShell:   $env:GATE_INSECURE=1
  Linux / mac:  export GATE_INSECURE=1
"""
import argparse
import json, math, os, ssl, time, sys, statistics
import urllib.parse, urllib.request, urllib.error
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

# ------------- 常量 -------------
GATE_BASE = "https://api.gateio.ws/api/v4"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "gate-quant-onefunc/1.3"
}

# ------------- SSL -------------
def _ssl_ctx():
    # 若你机器证书不对劲，可先临时设置：$env:GATE_INSECURE=1  再运行脚本
    if os.environ.get("GATE_INSECURE") == "1":
        try:
            return ssl._create_unverified_context()
        except Exception:
            pass
    return ssl.create_default_context()

# ------------- HTTP & 数据 -------------
def http_get(path: str, params: Dict[str,str], retries:int=3, timeout:int=20):
    """GET with basic retries; print server error body on HTTPError for debugging."""
    qs = urllib.parse.urlencode(params)
    url = f"{GATE_BASE}{path}?{qs}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS, method="GET")
            with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=timeout) as r:
                data = r.read()
                return json.loads(data.decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 打印服务端错误体，便于排查
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            print(f"[HTTP {e.code}] {url}\n{body}", file=sys.stderr)
            last = e
            # 4xx 多半是参数问题，不再重试
            if 400 <= e.code < 500:
                break
        except Exception as e:
            last = e
            # 逐次退避
            time.sleep(0.6 * (attempt + 1))
    raise last

def interval_to_seconds(interval: str) -> int:
    s = interval.lower().strip()
    if s.endswith("s"): return int(s[:-1])
    if s.endswith("m"): return int(s[:-1]) * 60
    if s.endswith("h"): return int(s[:-1]) * 3600
    if s.endswith("d"): return int(s[:-1]) * 86400
    raise ValueError(f"bad interval: {interval}")

def fetch_candles(symbol:str, interval:str, t_from:int, t_to:int) -> List[Dict]:
    """
    /spot/candlesticks 不接受 limit 参数。最多返回 1000 条，调用方分页。
    返回通常为: [t, v_quote, close, high, low, open]；少数环境会多带 base_vol/finished。
    """
    if t_from >= t_to:
        return []
    raw = http_get("/spot/candlesticks", {
        "currency_pair":symbol, "interval":interval, "from":t_from, "to":t_to
    })
    rows=[]
    for it in raw:
        ts   = int(float(it[0]))
        qv   = float(it[1]) if len(it)>1 else 0.0
        close= float(it[2]) if len(it)>2 else 0.0
        high = float(it[3]) if len(it)>3 else close
        low  = float(it[4]) if len(it)>4 else close
        open_= float(it[5]) if len(it)>5 else close
        bv   = 0.0
        # base_vol 可能在位置6，也可能没有
        if len(it) > 6:
            try:
                bv = float(it[6])
            except Exception:
                bv = qv
        else:
            bv = qv  # 兜底
        fin  = True  # 该接口一般不回 finished，这里简单当成已收盘
        rows.append({
            "ts": ts,
            "time": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
            "open": open_, "high": high, "low": low, "close": close,
            "base_vol": bv, "quote_vol": qv, "finished": fin
        })
    rows.sort(key=lambda x:x["ts"])
    return rows

def fetch_candles_window(symbol:str, interval:str, t_from:int, t_to:int) -> List[Dict]:
    """分页抓取整窗数据，按每段最多 1000 根（由时间跨度控制）。"""
    out=[]; sec_bar=interval_to_seconds(interval); cur=t_from
    # 防止过大窗口导致长时间请求
    hard_cap = 200000  # 最多抓20万根
    fetched = 0
    # Gate: 最多允许距今 10000 根以前的数据窗口。将起点钳制到这个范围内。
    cur = max(cur, t_to - sec_bar*9999)
    while cur < t_to and fetched < hard_cap:
        part=fetch_candles(symbol, interval, cur, min(t_to, cur+sec_bar*999))
        if not part: break
        out.extend(part)
        fetched += len(part)
        # 下一段窗口从最后一根的下一根开始
        cur=part[-1]["ts"] + sec_bar
    return out

# ------------- 指标 -------------
def ema(vals: List[float], span:int)->List[float]:
    if not vals: return []
    k=2/(span+1); out=[]; e=vals[0]
    for v in vals:
        e = v*k + e*(1-k); out.append(e)
    return out

def sma(vals: List[float], n:int)->List[float]:
    out=[]; s=0.0; q=[]
    for v in vals:
        q.append(v); s+=v
        if len(q)>n: s-=q.pop(0)
        out.append(s/max(1,len(q)))
    return out

def rsi(vals: List[float], period:int=14)->List[float]:
    if len(vals)<2: return [50.0]*len(vals)
    rsis=[50.0]; gains=[]; losses=[]
    for i in range(1, min(period+1,len(vals))):
        d=vals[i]-vals[i-1]; gains.append(max(d,0.0)); losses.append(max(-d,0.0))
    ag=sum(gains)/max(1,len(gains)); al=sum(losses)/max(1,len(losses))
    while len(rsis)<period:
        rs = (ag/(al+1e-12)) if al>0 else 9999.0
        rsis.append(100-100/(1+rs))
    for i in range(period+1,len(vals)):
        d=vals[i]-vals[i-1]; g=max(d,0.0); l=max(-d,0.0)
        ag=(ag*(period-1)+g)/period; al=(al*(period-1)+l)/period
        rs=(ag/(al+1e-12)) if al>0 else 9999.0
        rsis.append(100-100/(1+rs))
    return rsis[:len(vals)]

def atr(high:List[float], low:List[float], close:List[float], period:int=14)->List[float]:
    if not close: return []
    trs=[0.0]
    for i in range(1,len(close)):
        tr=max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        trs.append(tr)
    return ema(trs, period)

def macd(close: List[float], fast:int=12, slow:int=26, sig:int=9):
    ema_f=ema(close, fast); ema_s=ema(close, slow)
    dif=[(ema_f[i]-ema_s[i]) for i in range(len(close))]
    dea=ema(dif, sig)
    macd_bar=[2*(dif[i]-dea[i]) for i in range(len(dif))]
    return dif, dea, macd_bar

def boll(close: List[float], n:int=20, k:float=2.0):
    ma=sma(close, n)
    std=[]; buf=[]
    for v in close:
        buf.append(v)
        if len(buf)>n: buf.pop(0)
        m=sum(buf)/len(buf)
        var=sum((x-m)**2 for x in buf)/max(1,len(buf))
        std.append(max(var,0.0)**0.5)
    upper=[ma[i]+k*std[i] for i in range(len(close))]
    lower=[ma[i]-k*std[i] for i in range(len(close))]
    return ma, upper, lower

def zscore_last(series: List[float], lb:int)->float:
    if len(series)<lb: return 0.0
    w=series[-lb:]; m=sum(w)/lb
    var=sum((x-m)**2 for x in w)/max(1,lb-1)
    sd=max(var,1e-12)**0.5
    return (series[-1]-m)/(sd+1e-12)

def ols_slope_t(y: List[float])->Tuple[float,float]:
    n=len(y)
    if n<3: return 0.0,0.0
    xm=(n-1)/2.0; ym=sum(y)/n
    num=den=0.0
    for i in range(n):
        dx=i-xm; num+=dx*(y[i]-ym); den+=dx*dx
    slope=num/(den+1e-12)
    ss=0.0
    for i in range(n):
        yh=ym+slope*(i-xm); ss+=(y[i]-yh)**2
    sigma2=ss/max(1,n-2); se=(sigma2/(den+1e-12))**0.5
    return slope, slope/(se+1e-12)

# ------------- 交易决策 & 分析 -------------
def volume_ratio(vol: List[float], n:int)->float:
    if len(vol)<n: return float("nan")
    base=sum(vol[-n:])/n
    return vol[-1]/(base+1e-12)

def support_resistance(high:List[float], low:List[float], lookback:int=20)->Tuple[float,float,float]:
    """返回(近端压力, 近端支撑, pivot)"""
    if not high or not low:
        return float("nan"), float("nan"), float("nan")
    recent_high=max(high[-lookback:]) if len(high)>=lookback else max(high or [0])
    recent_low =min(low[-lookback:])  if len(low)>=lookback  else min(low or [0])
    H, L = high[-1], low[-1]
    C = (H+L)/2.0
    pivot=(H+L+C)/3.0
    return recent_high, recent_low, pivot

def decide_action(candles: List[Dict], sec_bar:int,
                  ema_fast:int=9, ema_slow:int=21,
                  breakout:int=10, vol_window:int=20, vol_thr:float=1.1,
                  z_look:int=12, z_thr:float=1.8, atr_p:int=14, atr_mult:float=0.6):
    """
    综合多指标给出信号：买多/做空/观望；并返回用于解释的上下文。
    """
    if not candles:
        return {"action":"观望","reason":"无数据","ctx":{}}
    need=max(ema_fast, ema_slow, breakout, vol_window, atr_p, z_look)+3
    if len(candles)<need:
        return {"action":"观望","reason":"样本不足","ctx":{}}

    close=[c["close"] for c in candles]
    high =[c["high"]  for c in candles]
    low  =[c["low"]   for c in candles]
    vol  =[c["base_vol"] for c in candles]

    ema_f=ema(close, ema_fast); ema_s=ema(close, ema_slow)
    rs  =rsi(close, 14)
    atr_arr=atr(high, low, close, atr_p)
    dif, dea, macd_bar = macd(close)
    ma, up, lo = boll(close, 20, 2.0)
    vr  =volume_ratio(vol, vol_window)
    z   =zscore_last(close, z_look)
    rh, rl, pv = support_resistance(high, low, lookback=20)

    last=close[-1]; prev=close[-2]
    ema_cross_up   = ema_f[-2] <= ema_s[-2] and ema_f[-1] > ema_s[-1]
    ema_cross_down = ema_f[-2] >= ema_s[-2] and ema_f[-1] < ema_s[-1]
    macd_up   = dif[-1] > dea[-1] and macd_bar[-1] > 0
    macd_down = dif[-1] < dea[-1] and macd_bar[-1] < 0
    near_res  = abs(last - rh) <= atr_mult*atr_arr[-1]
    near_sup  = abs(last - rl) <= atr_mult*atr_arr[-1]
    breakout_up   = last > max(high[-breakout:])
    breakout_down = last < min(low[-breakout:])

    long_score = 0
    short_score= 0
    reasons=[]

    if ema_cross_up:
        long_score += 1; reasons.append("EMA金叉")
    if ema_cross_down:
        short_score+= 1; reasons.append("EMA死叉")
    if macd_up:
        long_score += 1; reasons.append("MACD看多")
    if macd_down:
        short_score+= 1; reasons.append("MACD看空")
    if breakout_up:
        long_score += 1; reasons.append(f"{breakout}根内突破新高")
    if breakout_down:
        short_score+= 1; reasons.append(f"{breakout}根内跌破新低")
    if vr>=vol_thr:
        # 放量配合方向
        if last>ma[-1]: long_score+=1; reasons.append("放量上行")
        if last<ma[-1]: short_score+=1; reasons.append("放量下行")
    if z>=z_thr:
        long_score+=1; reasons.append(f"Z分数{z:.2f}偏热")
    if z<=-z_thr:
        short_score+=1; reasons.append(f"Z分数{z:.2f}偏冷")

    # 价位贴近支撑/压力时，倾向反转
    if near_res:
        short_score+=1; reasons.append("贴近压力位")
    if near_sup:
        long_score+=1; reasons.append("贴近支撑位")

    # RSI 极值辅助（不过度使用）
    if rs[-1] >= 70:
        short_score+=1; reasons.append("RSI高位回落风险")
    if rs[-1] <= 30:
        long_score+=1; reasons.append("RSI低位反弹机会")

    # 结合ATR给出大致止损
    stop_long  = last - atr_mult*atr_arr[-1]
    stop_short = last + atr_mult*atr_arr[-1]

    action = "观望"
    if long_score >= short_score+1 and long_score >= 2:
        action="做多"
    elif short_score >= long_score+1 and short_score >= 2:
        action="做空"

    ctx={
        "price": last, "ema_fast": ema_f[-1], "ema_slow": ema_s[-1],
        "macd_dif": dif[-1], "macd_dea": dea[-1], "macd_bar": macd_bar[-1],
        "rsi": rs[-1], "boll_ma": ma[-1], "boll_up": up[-1], "boll_lo": lo[-1],
        "atr": atr_arr[-1], "vr": vr, "z": z,
        "resistance": rh, "support": rl, "pivot": pv,
        "stop_long": stop_long, "stop_short": stop_short,
        "long_score": long_score, "short_score": short_score,
    }
    reason = " | ".join(reasons) if reasons else "信号弱/互相矛盾"
    return {"action":action, "reason":reason, "ctx":ctx}

# ------------- 预测（简单线性外推） -------------
def predict_next_seconds(candles: List[Dict], seconds_ahead:int=3)->Tuple[float,float]:
    """
    用 OLS 对最近若干收盘价做线性拟合，外推 seconds_ahead 秒后的价格。
    返回 (pred_price, slope_per_sec)
    """
    if not candles: return float("nan"), 0.0
    # 取近 N 根（>=8），对应的“索引”为时间戳，做时间权重更合理
    N = min(36, max(8, len(candles)))
    recent = candles[-N:]
    xs = [c["ts"] for c in recent]
    ys = [c["close"] for c in recent]
    # 标准化时间，避免大数精度
    t0 = xs[0]
    xs_n = [x - t0 for x in xs]
    # 计算斜率与截距
    n = len(xs_n)
    sx = sum(xs_n); sy = sum(ys)
    sxx = sum(x*x for x in xs_n)
    sxy = sum(xs_n[i]*ys[i] for i in range(n))
    den = (n*sxx - sx*sx) if n>1 else 0.0
    if abs(den) < 1e-12:
        return ys[-1], 0.0
    b = (n*sxy - sx*sy) / den      # slope per second（因为xs单位是秒）
    a = (sy - b*sx)/n              # intercept
    pred = a + b*(xs_n[-1] + seconds_ahead)
    return pred, b

# ------------- 回测 -------------
def backtest(candles: List[Dict], sec_bar:int,
             fee_rate:float=0.0006,  # 单边万6
             slip:float=0.5          # 滑点（USD）
             )->Dict:
    """
    简单规则：当 decide_action 给出“做多/做空”且分数优势>=1时入场，反向或观望退出。
    仅做演示，不代表实盘效果。
    """
    if len(candles) < 50:
        return {"trades":0,"pnl":0.0,"win_rate":0.0,"detail":[]}

    pos=0   # 0 无仓, 1 多, -1 空
    entry=0.0
    pnl=0.0
    wins=0; trades=0
    detail=[]

    for i in range(50, len(candles)):
        sub = candles[:i+1]
        sig = decide_action(sub, sec_bar)
        price = sub[-1]["close"]
        # 出场条件：若有仓且信号非本方向
        if pos==1 and sig["action"]!="做多":
            # 平多
            exit_p = price - slip
            pnl += (exit_p - entry) - fee_rate*(exit_p+entry)
            wins += 1 if exit_p>entry else 0
            trades += 1
            detail.append(("平多", sub[-1]["time"], exit_p, pnl))
            pos=0
        elif pos==-1 and sig["action"]!="做空":
            # 平空
            exit_p = price + slip
            pnl += (entry - exit_p) - fee_rate*(exit_p+entry)
            wins += 1 if exit_p<entry else 0
            trades += 1
            detail.append(("平空", sub[-1]["time"], exit_p, pnl))
            pos=0

        # 入场条件：无仓，且信号明确
        if pos==0 and sig["action"] in ("做多","做空"):
            if sig["action"]=="做多":
                entry = price + slip
                pos=1
                detail.append(("开多", sub[-1]["time"], entry, pnl))
            else:
                entry = price - slip
                pos=-1
                detail.append(("开空", sub[-1]["time"], entry, pnl))

    win_rate = (wins/trades) if trades>0 else 0.0
    return {"trades":trades, "pnl":round(pnl,4), "win_rate":round(win_rate,3), "detail":detail}

# ------------- 实盘轮询 -------------
def fetch_recent(symbol:str, interval:str, bars:int)->List[Dict]:
    sec_bar = interval_to_seconds(interval)
    now = int(time.time())
    t_to   = now
    t_from = now - bars*sec_bar - 1
    return fetch_candles_window(symbol, interval, t_from, t_to)

def print_signal(sig:Dict, pred:Tuple[float,float], sec_bar:int):
    price = sig["ctx"].get("price", float("nan"))
    pred_price, slope = pred
    slope_per_bar = slope * sec_bar
    ctx = sig["ctx"]
    ts_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = (f"[{ts_str} UTC] 价:{price:.2f}  信号:{sig['action']}  因素:{sig['reason']} | "
            f"RSI:{ctx.get('rsi',float('nan')):.1f}  Z:{ctx.get('z',float('nan')):.2f}  "
            f"VR:{ctx.get('vr',float('nan')):.2f}  ATR:{ctx.get('atr',float('nan')):.2f} | "
            f"支撑:{ctx.get('support',float('nan')):.2f}  压力:{ctx.get('resistance',float('nan')):.2f} | "
            f"3秒预测:{pred_price:.2f}  斜率/秒:{slope:.4f}  斜率/一根:{slope_per_bar:.4f}")
    print(line)

def cmd_backtest(args):
    symbol=args.symbol; interval=args.interval; days=args.days
    sec_bar = interval_to_seconds(interval)
    t_to = int(time.time())
    t_from = t_to - int(days*24*3600)
    candles = fetch_candles_window(symbol, interval, t_from, t_to)
    if not candles:
        print("没有拿到K线数据，请检查网络或参数。")
        return
    print(f"已获取 {len(candles)} 根K线（{interval}），时间范围 {candles[0]['time']} 至 {candles[-1]['time']} (UTC)")
    sig = decide_action(candles, sec_bar)
    pred = predict_next_seconds(candles, seconds_ahead=3)
    print_signal(sig, pred, sec_bar)
    # 回测
    bt = backtest(candles, sec_bar)
    print(f"回测交易数: {bt['trades']}  胜率: {bt['win_rate']:.2%}  累计PnL(单位价差): {bt['pnl']}")
    if args.verbose and bt["detail"]:
        for act,t,p,cur in bt["detail"][-20:]:
            print(f"  {act} @ {t}  价:{p:.2f}  累计PnL:{cur:.4f}")

def cmd_live(args):
    symbol=args.symbol; interval=args.interval
    sec_bar = interval_to_seconds(interval)
    # 先预热抓一段，保证指标稳定
    candles = fetch_recent(symbol, interval, bars=max(60, 6*max(21, 26)))
    if not candles:
        print("没有拿到K线数据，请检查网络或参数。")
        return
    print(f"预热完成。当前已有 {len(candles)} 根K线，最新时间 {candles[-1]['time']} (UTC)")
    last_ts_seen = candles[-1]["ts"]
    while True:
        try:
            # 只刷新近若干根
            more = fetch_recent(symbol, interval, bars=120)
            if more and more[-1]["ts"] != last_ts_seen:
                candles = more  # 替换为最新窗口
                last_ts_seen = candles[-1]["ts"]
                sig = decide_action(candles, sec_bar)
                pred = predict_next_seconds(candles, seconds_ahead=3)
                print_signal(sig, pred, sec_bar)
            time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n已退出。")
            break
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            time.sleep(1.5)



def cmd_combo(args):
    """
    先做历史回测 + 当前信号与3秒预测，然后无缝进入实盘每秒轮询（同 live）。
    可用 --live-seconds 指定轮询时长（秒），默认0为无限直到 Ctrl+C。
    """
    symbol=args.symbol; interval=args.interval; days=args.days; live_seconds=int(getattr(args, "live_seconds", 0) or 0)
    sec_bar = interval_to_seconds(interval)
    # 历史段
    t_to = int(time.time())
    t_from = t_to - int(days*24*3600)
    candles = fetch_candles_window(symbol, interval, t_from, t_to)
    if not candles:
        print("没有拿到K线数据，请检查网络或参数。")
        return
    print(f"[COMBO] 历史段：已获取 {len(candles)} 根K线（{interval}），范围 {candles[0]['time']} 至 {candles[-1]['time']} (UTC)")
    sig = decide_action(candles, sec_bar)
    pred = predict_next_seconds(candles, seconds_ahead=3)
    print_signal(sig, pred, sec_bar)
    bt = backtest(candles, sec_bar)
    print(f"[COMBO] 回测结果 -> 交易数: {bt['trades']}  胜率: {bt['win_rate']:.2%}  累计PnL: {bt['pnl']}")
    # 进入实盘轮询
    print("[COMBO] 进入实盘轮询（每秒）... 按 Ctrl+C 退出。")
    start_time = time.time()
    last_ts_seen = candles[-1]["ts"]
    while True:
        try:
            more = fetch_recent(symbol, interval, bars=120)
            if more and more[-1]["ts"] != last_ts_seen:
                candles = more
                last_ts_seen = candles[-1]["ts"]
                sig = decide_action(candles, sec_bar)
                pred = predict_next_seconds(candles, seconds_ahead=3)
                print_signal(sig, pred, sec_bar)
            time.sleep(1.0)
            if live_seconds>0 and (time.time() - start_time) >= live_seconds:
                print("[COMBO] 达到指定轮询时长，退出。")
                break
        except KeyboardInterrupt:
            print("\n[COMBO] 已退出。")
            break
        except Exception as e:
            print(f"[COMBO][ERROR] {e}")
            time.sleep(1.5)


def build_parser():
    p = argparse.ArgumentParser(description="Gate.io 历史回测 + 指标分析 + 实盘轮询 + 3秒预测 (纯标准库)")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_bt = sub.add_parser("backtest", help="历史回测 + 当前信号")
    p_bt.add_argument("--symbol", default="ETH_USDT")
    p_bt.add_argument("--interval", default="10s")
    p_bt.add_argument("--days", type=float, default=2.0, help="历史区间天数")
    p_bt.add_argument("-v","--verbose", action="store_true")
    p_bt.set_defaults(func=cmd_backtest)

    p_lv = sub.add_parser("live", help="实盘每秒轮询 + 3秒预测")
    p_lv.add_argument("--symbol", default="ETH_USDT")
    p_lv.add_argument("--interval", default="10s")
    p_lv.set_defaults(func=cmd_live)

    p_cb = sub.add_parser("combo", help="先回测再进入实盘轮询")
    p_cb.add_argument("--symbol", default="ETH_USDT")
    p_cb.add_argument("--interval", default="10s")
    p_cb.add_argument("--days", type=float, default=2.0, help="历史区间天数")
    p_cb.add_argument("--live-seconds", type=int, default=0, help="轮询秒数(0=无限)")
    p_cb.set_defaults(func=cmd_combo)
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
