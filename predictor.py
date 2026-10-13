"""منطق التنبؤ: نفس حساب المؤشرات المستخدم في التدريب (common.py)، وقراءة الموديل من ملف npz بدون sklearn."""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE / "model" / "BTC_30m_model_058.npz"))
THRESHOLD = 0.58
PAYOUT = 0.83
BREAK_EVEN = 1 / (1 + PAYOUT)
HORIZON_MIN = 30
HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
]


class NpModel:
    """يقرأ أشجار HistGradientBoosting من ملف npz (نفس الأرقام تماماً)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.features = [str(x) for x in z["features"]]
        self.off = z["offsets"]
        self.base = float(z["baseline"])
        self.fi, self.thr = z["feature_idx"], z["thr"]
        self.l, self.r, self.val = z["left"], z["right"], z["value"]
        self.ml, self.leaf = z["miss_left"], z["is_leaf"]
        self.n_trees = len(self.off) - 1

    def p_up(self, x):
        raw = self.base
        for t in range(self.n_trees):
            o = self.off[t]
            n = 0
            while not self.leaf[o + n]:
                k = o + n
                v = x[self.fi[k]]
                go_left = (self.ml[k] == 1) if np.isnan(v) else (v <= self.thr[k])
                n = self.l[k] if go_left else self.r[k]
            raw += self.val[o + n]
        return 1.0 / (1.0 + np.exp(-raw))


_model = None


def get_model():
    global _model
    if _model is None:
        _model = NpModel(MODEL_PATH)
    return _model


def build_features(df):
    """نفس منطق common.build_features بالضبط."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "n_trades", "buy_vol", "sell_vol", "quote_vol", "vwap"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    close = df["close"].astype(float)
    f = pd.DataFrame(index=df.index)
    for n in [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]:
        f[f"ret_{n}"] = np.log(close / close.shift(n))
    r = np.log(close / close.shift(1))
    for n in [5, 15, 30, 60, 120, 240]:
        f[f"vol_{n}"] = r.rolling(n).std()
    buy = df["buy_vol"].astype(float)
    sell = df["sell_vol"].astype(float)
    ofi = (buy - sell) / (buy + sell + 1e-12)
    for n in [1, 3, 5, 15, 30, 60, 240]:
        f[f"ofi_{n}"] = ofi.rolling(n).mean()
    f["log_n_trades"] = np.log1p(df["n_trades"].clip(lower=0))
    f["log_quote_vol"] = np.log1p(df["quote_vol"].clip(lower=0))
    f["vwap_dev"] = (close - df["vwap"].astype(float)) / (df["vwap"].astype(float) + 1e-12)
    f["range_pct"] = (df["high"] - df["low"]) / (close + 1e-12)
    f["body_pct"] = (df["close"] - df["open"]) / (close + 1e-12)
    f["close_position"] = (df["close"] - df["low"]) / ((df["high"] - df["low"]) + 1e-12)
    m = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    f["hour_sin"] = np.sin(2 * np.pi * m / 1440)
    f["hour_cos"] = np.cos(2 * np.pi * m / 1440)
    f["dow"] = df["ts"].dt.dayofweek
    return df, f.replace([np.inf, -np.inf], np.nan)


def predict_frame(candles):
    """يرجع جدول توقعات لكل شمعة لها مؤشرات كاملة."""
    model = get_model()
    d, F = build_features(candles)
    F = F[model.features].replace([np.inf, -np.inf], np.nan)
    valid = F.dropna()
    rows = []
    for i in valid.index:
        p = float(model.p_up(valid.loc[i].values.astype(float)))
        rows.append({
            "ts": d.loc[i, "ts"],
            "price": float(d.loc[i, "close"]),
            "p_up": p,
            "confidence": max(p, 1 - p),
            "direction": "صعود" if p >= 0.5 else "هبوط",
            "signal": max(p, 1 - p) >= THRESHOLD,
        })
    return pd.DataFrame(rows)


def fetch_candles(limit=300, timeout=20):
    """آخر شموع 1 دقيقة مغلقة من Binance (نفس الطريقة المستخدمة في الداشبورد)."""
    rows, err = None, None
    for h in HOSTS:
        try:
            r = requests.get(
                h + "/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit},
                timeout=timeout,
            )
            r.raise_for_status()
            rows = r.json()
            break
        except Exception as e:  # noqa: BLE001
            err = e
    if rows is None:
        raise RuntimeError(f"تعذر جلب البيانات من Binance: {err!r}")
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
            "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    raw = pd.DataFrame(rows, columns=cols)
    raw = raw[raw["close_time"].astype("int64") < int(time.time() * 1000)].copy()  # الشموع المغلقة فقط
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "n_trades", "taker_buy_base"]:
        raw[c] = pd.to_numeric(raw[c])
    d = pd.DataFrame({
        "ts": pd.to_datetime(raw["open_time"], unit="ms", utc=True),
        "open": raw["open"], "high": raw["high"], "low": raw["low"], "close": raw["close"],
        "n_trades": raw["n_trades"], "buy_vol": raw["taker_buy_base"],
        "sell_vol": (raw["volume"] - raw["taker_buy_base"]).clip(lower=0),
        "quote_vol": raw["quote_volume"],
    })
    d["vwap"] = d["quote_vol"] / (d["buy_vol"] + d["sell_vol"] + 1e-12)
    return d


_cache = {"t": 0.0, "data": None}


def live_snapshot(history=30, ttl=10):
    """يرجع آخر توقع + آخر N توقعات. مخزّن مؤقتاً لمدة ttl ثانية لتخفيف الطلبات."""
    now = time.time()
    if _cache["data"] is not None and now - _cache["t"] < ttl:
        return _cache["data"]
    candles = fetch_candles(limit=300)
    table = predict_frame(candles)
    if table.empty:
        raise RuntimeError("لا توجد شموع كافية لحساب المؤشرات")
    last = table.iloc[-1]
    hist = table.tail(history).iloc[::-1]
    data = {
        "fetched_at": pd.Timestamp.utcnow().isoformat(),
        "candle_time": str(last["ts"]),
        "price": float(last["price"]),
        "p_up": float(last["p_up"]),
        "confidence": float(last["confidence"]),
        "direction": last["direction"],
        "signal": bool(last["signal"]),
        "threshold": THRESHOLD,
        "horizon_min": HORIZON_MIN,
        "break_even": BREAK_EVEN,
        "history": [
            {"ts": str(r.ts), "price": float(r.price), "p_up": float(r.p_up), "direction": r.direction,
             "confidence": float(r.confidence), "signal": bool(r.signal)}
            for r in hist.itertuples()
        ],
    }
    _cache.update(t=now, data=data)
    return data
