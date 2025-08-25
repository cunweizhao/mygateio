#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate.io 一体化：历史回测 + 指标分析(BOLL/MACD/量能/支撑压力) + 实盘每秒轮询 + 3秒预测
纯标准库，无第三方依赖。Python 3.8+
"""

import json, math, os, ssl, time, urllib.parse, urllib.request
from datetime import datetime
from typing import List, Dict, Tuple

# ====== 可调开关 ======
ALLOW_INSECURE_SSL = False  # 如遇本机证书错误，可暂时设为 True（仅调试用）
DEFAULT_INTERVAL   = "1m"   # 默认用 1m 更稳；要更细可改 "10s"（Gate 文档支持 10s 间隔）
# =====================

GATE_BASES = [
    os.environ.get("GATE_BASE") or "https://api.gateio.ws",
    "https://api.gateeu.com",  # 官方文档给的 EU 域
]
HEADERS = {
    "Accept":"application/json",
    "Content-Type":"application/json",
    "User-Agent":"gate-quant-onefunc/1.2"
}

# ======================= HTTP & 数据 =======================

def _ssl_context():
    if ALLOW_INSECURE_SSL:
        ctx = ssl._create_unverified_context()
    else:
        ctx = ssl.create_default_context()
    return ctx

def http_get(path: str, params: Dict[str,str], retries:int=2, timeout:int=20):
    """轮询多个 BASE，支持重试；4xx 打印返回体便于排查。"""
    qs = urllib.parse.urlencode(params)
    last = None
    for base in GATE_BASES:
        url = f"{base}/api/v4{path}?{qs}"
        for attempt in range(retries+1):
            try:
                req = urllib.request.Request(url, headers=HEADERS, method="GET")
                with urllib.request.urlopen(req, context=_ssl_context(), timeout=timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode("utf-8", errors="ignore")
                except Exception:
                    body = ""
                print(f"[HTTP {e.code}] {url}\n{body}\n")
                last = e
                # 4xx 多为参数或时间窗问题，不再对同一 BASE 重试
                if 400 <= e.code < 500:
                    break
            except Exception as e:
                last = e
                time.sleep(0.6*(attempt+1))
        # 换下一个 BASE 继续
    raise last

def interval_to_seconds(interval: str) -> int:
    s = interval.lower().strip()
    if s.endswith("s"): return int(s[:-1])
    if s.endswith("m"): return int(s[:-1]) * 60
    if s.endswith("h"): return int(s[:-1]) * 3600
    if s.endswith("d"): return int(s[:-1]) * 86400
    raise ValueError("bad interval")

def fetch_candles(symbol:str, interval:str, t_from:int, t_to:int) -> List[Dict]:
    """
    /spot/candlesticks 不接受 limit 参数；最多返回 1000 条，调用方按时间窗分页。
    返回行一般为:
      [ts, quote_vol, close, high, low, open, base_vol, finished]
    """
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
        bv   = float(it[6]) if len(it)>6 else qv
        fin  = True
        if len(it)>7:
            try:
                fin = str(it[7]).lower()=="true"
            except Exception:
                fin = True
        rows.append({
            "ts": ts,
            "time": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
            "open": open_, "high": high, "low": low, "close": close,
            "base_vol": bv, "quote_vol": qv, "finished": fin
        })
    rows.sort(key=lambda x:x["ts"])
    return rows

def fetch_candles_window(symbol:str, interval:str, t_from:int, t_to:int) -> List[Dict]:
    """按 1000 根一步分页抓取整窗。"""
    out=[]; sec_bar=interval_to_seconds(interval); cur=t_from
    # 防呆：from 必须 < to
    if t_from >= t_to: return out
    while cur < t_to:
        part=fetch_candles(symbol, interval, cur, min(t_to, cur+sec_bar*999))
        if not part: break
        out.extend(part)
        cur=part[-1]["ts"] + sec_bar
    return out

# ======================= 指标 =======================

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
        std.append(var**0.5)
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

# ======================= 交易决策 & 预测 =======================

def volume_ratio(vol: List[float], n:int)->float:
    if len(vol)<n: return float("nan")
    base=sum(vol[-n:])/n
    return vol[-1]/(base+1e-12)

def support_resistance(high:List[float], low:List[float], lookback:int=20)->Tuple[float,float,float]:
    """返回(近端压力, 近端支撑, pivot)"""
    recent_high=max(high[-lookback:]) if len(high)>=lookback else max(high or [0])
    recent_low =min(low[-lookback:])  if len(low)>=lookback else min(low or [0])
    H, L = high[-1], low[-1]
    C = (H+L)/2.0
    pivot=(H+L+C)/3.0
    return recent_high, recent_low, pivot

def decide_action(candles: List[Dict], sec_bar:int,
                  ema_fast:int=9, ema_slow:int=21,
                  breakout:int=10, vol_window:int=20, vol_thr:float=1.1,
                  z_look:int=12, z_thr:float=1.8, atr_p:int=14, atr_mult:float=0.6):
    if not candles: return {"action":"观望","reason":"无数据","ctx":{}}
    if not candles[-1].get("finished", True): candles=candles[:-1]
    need=max(ema_fast, ema_slow, breakout, vol_window, atr_p, z_look)+1
    if len(candles)<need: return {"action":"观望","reason":"样本不足","ctx":{}}

    close=[c["close"] for c in candles]
    high =[c["high"]  for c in candles]
    low  =[c["low"]   for c in candles]
    vol  =[c["base_vol"] for c in candles]

    ema_f=ema(close, ema_fast); ema_s=ema(close, ema_slow)
    rs= rsi(close, 14)
    atr_arr=atr(high, low, close, atr_p)
    dif, dea, macd_bar = macd(close, 12, 26, 9)
    ma, up, lo = boll(close, 20, 2.0)

    vr=volume_ratio(vol, vol_window)
    vol_regime = "放量" if (vr==vr and vr>=1.1) else ("收量" if (vr==vr and vr<=0.9) else "正常")

    near_res, near_sup, pivot = support_resistance(high, low, lookback=breakout)

    price=close[-1]
    trend_up = ema_f[-1]>ema_s[-1]
    trend_dn = ema_f[-1]<ema_s[-1]
    recent_high=max(high[-breakout:]); recent_low=min(low[-breakout:])
    allow_novol = (vol_thr<=1.0 or not (vr==vr))
    bo_long = trend_up and price>recent_high and (allow_novol or vr>=vol_thr)
    bo_short= trend_dn and price<recent_low and (allow_novol or vr>=vol_thr)

    rets=[0.0]+[(close[i]-close[i-1])/(close[i-1]+1e-12) for i in range(1,len(close))]
    z = zscore_last(rets, min(z_look, max(5, z_look)))
    impulse_dn = (price <= close[-3] - atr_mult*(atr_arr[-1] or 0)) and trend_dn
    impulse_up = (price >= close[-3] + atr_mult*(atr_arr[-1] or 0)) and trend_up
    mo_short = (z<=-z_thr) or impulse_dn
    mo_long  = (z>= z_thr) or impulse_up

    boll_bias = "上轨" if price>=up[-1] else ("下轨" if price<=lo[-1] else "中轨")
    macd_bias_pos = (dif[-1]>0 and macd_bar[-1]>0)
    macd_bias_neg = (dif[-1]<0 and macd_bar[-1]<0)

    reasons=[]; action="观望"
    if (mo_short or bo_short) and (macd_bias_neg or price<ma[-1] or boll_bias=="上轨"):
        action="做空"; reasons.append("动量/跌破+MACD空/均线下/上轨回落")
    if action=="观望" and (mo_long or bo_long) and (macd_bias_pos or price>ma[-1] or boll_bias=="下轨"):
        action="做多"; reasons.append("动量/突破+MACD多/均线上/下轨反弹")

    last=candles[-1]; o,h,l,c=last["open"], last["high"], last["low"], last["close"]
    upper=h-max(o,c); lower=min(o,c)-l; body=abs(c-o)
    if action=="做多" and upper>2*body: action="观望"; reasons.append("上影异常→观望")
    if action=="做空" and lower>2*body: action="观望"; reasons.append("下影异常→观望")

    ctx={
        "price":price, "ema_fast":ema_f[-1], "ema_slow":ema_s[-1], "rsi14":rs[-1],
        "atr":atr_arr[-1], "vratio":vr, "vol_regime":vol_regime,
        "recent_high":recent_high, "recent_low":recent_low,
        "near_res":near_res, "near_sup":near_sup, "pivot":pivot,
        "boll_mid":ma[-1], "boll_up":up[-1], "boll_lo":lo[-1],
        "macd_dif":dif[-1], "macd_dea":dea[-1], "macd_bar":macd_bar[-1],
        "zscore":z
    }
    return {"action":action, "reason":"; ".join(reasons) if reasons else "无触发", "ctx":ctx}

def forecast_next(candles: List[Dict], sec_bar:int, horizon_secs:int=3) -> Dict:
    if not candles: return {"dir":"flat","up":0.5,"dn":0.5,"explain":"无数据"}
    if not candles[-1].get("finished", True): candles=candles[:-1]
    if len(candles)<10: return {"dir":"flat","up":0.5,"dn":0.5,"explain":"样本不足"}

    close=[c["close"] for c in candles]
    rsi14=rsi(close,14)
    hb=max(1, round(horizon_secs/max(1,sec_bar)))
    lb=max(5, 5*hb)
    tail=close[-lb:]

    _,t=ols_slope_t(tail)
    rets=[0.0]+[(close[i]-close[i-1])/(close[i-1]+1e-12) for i in range(1,len(close))]
    z=zscore_last(rets, min(len(rets), max(5, lb//2)))
    ema_f=ema(close,9); ema_s=ema(close,21)
    gap=(ema_f[-1]-ema_s[-1])/(close[-1]+1e-12)
    _,rt=ols_slope_t(rsi14[-min(lb, len(rsi14)):])

    def sig(x): return 1/(1+math.exp(-x))
    p1=sig(t); p2=sig(2.5*z); p3=sig(30*gap); p4=sig(rt/2.0)
    up=max(0.0, min(1.0, 0.4*p1+0.3*p2+0.2*p3+0.1*p4))
    dn=1.0-up
    if up>=0.55: d="up"
    elif dn>=0.55: d="down"
    else: d="flat"
    coarse = "" if horizon_secs>=sec_bar else f" (h={horizon_secs}s < bar={sec_bar}s)"
    return {"dir":d,"up":up,"dn":dn,
            "explain":f"Slope-t={t:.2f}, Z={z:.2f}, EMAGap={gap:.4f}, RSI-t={rt:.2f}{coarse}"}

# ======================= 回测 =======================

def backtest_fixed_hold(candles: List[Dict], sec_bar:int, hold_secs:int=120,
                        fee_bp:float=5.0, slip_bp:float=5.0) -> Dict:
    if not candles: return {"trades":[],"summary":{}}
    c=[x for x in candles if x.get("finished",True)]
    trades=[]; i=30
    while i < len(c)-2:
        win=c[:i+1]
        sig=decide_action(win, sec_bar)
        if sig["action"] in ("做多","做空"):
            entry=i+1; hold=max(1, math.ceil(hold_secs/sec_bar)); exit=min(len(c)-1, entry+hold)
            epx=c[entry]["open"]; xpx=c[exit]["open"]; dire= 1 if sig["action"]=="做多" else -1
            cost=(fee_bp+slip_bp)/10000.0*2
            raw=dire*(xpx/epx-1.0); net=raw - cost
            trades.append({"entry_time":c[entry]["time"], "exit_time":c[exit]["time"],
                           "side":sig["action"], "entry":epx, "exit":xpx, "net_ret":net,
                           "reason":sig["reason"]})
            i=exit+1
        else:
            i+=1
    rets=[t["net_ret"] for t in trades]
    win=sum(1 for r in rets if r>0); cum=1.0; eq=[]; mdd=0.0; peak=1.0
    for r in rets:
        cum*=(1+r); eq.append(cum); peak=max(peak,cum); mdd=max(mdd,(peak-cum)/peak)
    avg=sum(rets)/len(rets) if rets else 0.0
    var=sum((x-avg)**2 for x in rets)/max(1,len(rets)-1); sd=max(var,1e-12)**0.5
    sharpe=(avg/sd*math.sqrt(252*24*60*60/max(1,hold_secs))) if sd>0 else 0.0
    return {"trades":trades, "summary":{
        "trades":len(rets),
        "win_rate": win/max(1,len(rets)),
        "avg_net_ret":avg, "cum_return":cum-1.0, "max_drawdown":mdd, "sharpe_like":sharpe
    }}

# ======================= 自检 + 单方法入口 =======================

def _selfcheck(symbol:str, interval:str="1m"):
    """连通性快速自检：抓 5 分钟 1mK，失败会打印详细错误。"""
    try:
        sec_bar=interval_to_seconds(interval)
        t2=int(time.time())-2
        t1=t2-5*60
        data=fetch_candles_window(symbol, interval, t1, t2)
        if not data:
            print("[SELFTEST] 无法获取K线，请检查网络/时间窗/交易对。")
        else:
            last=data[-1]
            print(f"[SELFTEST] OK: {symbol} {interval} 条目={len(data)} 最新收盘={last['close']}")
    except Exception as e:
        print(f"[SELFTEST] 失败: {type(e).__name__}: {e}")

def run_quant(symbol:str="ETH_USDT", interval:str=DEFAULT_INTERVAL,
              hist_days:int=2, window_min:int=30,
              hold_secs:int=120, poll_secs:int=1, forecast_secs:int=3):
    # 自检
    _selfcheck(symbol, "1m")
    sec_bar=interval_to_seconds(interval)

    # --- 历史回测 ---
    to_ts=int(time.time())-3
    from_ts=to_ts - max(1, hist_days)*86400
    candles=fetch_candles_window(symbol, interval, from_ts, to_ts)
    bt=backtest_fixed_hold(candles, sec_bar, hold_secs=hold_secs, fee_bp=5.0, slip_bp=5.0)
    s=bt["summary"] or {}
    print("=== 回测结果 ===")
    print(f"交易笔数: {s.get('trades',0)}  胜率: {s.get('win_rate',0):.2%}")
    print(f"平均净收益/笔: {s.get('avg_net_ret',0):.4%}  累计收益: {s.get('cum_return',0):.2%}")
    print(f"最大回撤: {s.get('max_drawdown',0):.2%}  Sharpe-like: {s.get('sharpe_like',0):.2f}")
    print("================\n")

    # --- 实盘轮询 ---
    print(f"[LIVE] {symbol} interval={interval} poll={poll_secs}s window={window_min}m hold={hold_secs}s fcast={forecast_secs}s")
    while True:
        try:
            t2=int(time.time())-2
            t1=t2 - window_min*60
            win_candles=fetch_candles_window(symbol, interval, t1, t2)
            if not win_candles:
                print(f"[{datetime.now()}] 无数据")
            else:
                sig=decide_action(win_candles, sec_bar)
                fc =forecast_next(win_candles, sec_bar, horizon_secs=forecast_secs)
                ex=sig["ctx"]
                now_side=sig["action"]
                next_side = "做多" if fc["dir"]=="up" else ("做空" if fc["dir"]=="down" else "观望")
                vr=ex.get("vratio", float("nan"))
                print(
                    f"[{datetime.now()}] {symbol} | 现在建议: {now_side} ({sig['reason']}) | "
                    f"P={ex.get('price', float('nan')):.4f}  EMA9/21={ex.get('ema_fast',0):.4f}/{ex.get('ema_slow',0):.4f}  "
                    f"MACD(dif/dea/bar)={ex.get('macd_dif',0):.5f}/{ex.get('macd_dea',0):.5f}/{ex.get('macd_bar',0):.5f}  "
                    f"BOLL(mid/up/lo)={ex.get('boll_mid',0):.4f}/{ex.get('boll_up',0):.4f}/{ex.get('boll_lo',0):.4f}  "
                    f"量能VR={vr if vr==vr else float('nan'):.2f}({ex.get('vol_regime','?')})  "
                    f"支撑/压力≈ {ex.get('near_sup',0):.2f}/{ex.get('near_res',0):.2f}  "
                    f"| [下{forecast_secs}s预测] {next_side}  up={fc['up']:.2%} dn={fc['dn']:.2%} ({fc['explain']})"
                )
        except Exception as e:
            print(f"异常: {type(e).__name__}: {e}")
        finally:
            time.sleep(max(1, poll_secs))

# 直接脚本运行时的默认入口
if __name__ == "__main__":
    # 若你要 10s K，把 DEFAULT_INTERVAL 改为 "10s" 或这里传 interval="10s"
    run_quant(symbol="ETH_USDT", interval=DEFAULT_INTERVAL,
              hist_days=0.5, window_min=30,
              hold_secs=120, poll_secs=1, forecast_secs=3)
