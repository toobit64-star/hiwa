import os
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests
import pandas as pd
import ccxt
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("hiwacrypto-v2")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "").strip()
CHANNEL = os.getenv("CHANNEL_USERNAME", "@hiwacrypto").strip()
BASE_URL = os.getenv("BITGET_BASE_URL", "https://api.bitget.com").rstrip("/")

MARKETS = [x.strip().upper() for x in os.getenv("MARKETS", "SPOT,FUTURES").split(",") if x.strip()]
SYMBOLS = [x.strip().upper() for x in os.getenv(
    "SCAN_SYMBOLS",
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,DOTUSDT,LINKUSDT,TONUSDT,TRXUSDT"
).split(",") if x.strip()]

PRIMARY_TF = os.getenv("PRIMARY_TIMEFRAME", "1H").upper()
CONFIRM_TF = os.getenv("CONFIRM_TIMEFRAME", "4H").upper()
SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MINUTES", "15"))
MIN_SCORE = int(os.getenv("MIN_SCORE", "70"))
BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "20"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
MAX_PENDING = int(os.getenv("MAX_PENDING_SIGNALS", "20"))

# Growth scanner settings
GROWTH_ENABLED = os.getenv("GROWTH_SCANNER_ENABLED", "true").lower() == "true"
GROWTH_INTERVAL_MIN = int(os.getenv("GROWTH_SCAN_INTERVAL_MINUTES", "30"))
GROWTH_UNIVERSE = int(os.getenv("GROWTH_UNIVERSE", "35"))
GROWTH_RESULTS = int(os.getenv("GROWTH_RESULTS", "5"))
GROWTH_MIN_TURNOVER = float(os.getenv("GROWTH_MIN_TURNOVER_24H", "1000000"))
GROWTH_MIN_SCORE = int(os.getenv("GROWTH_MIN_SCORE", "65"))
# Multi-exchange scanner (public market data; no API keys required)
MULTI_EXCHANGE_ENABLED = os.getenv("MULTI_EXCHANGE_ENABLED", "true").lower() == "true"
EXCHANGES = [x.strip().lower() for x in os.getenv("EXCHANGES", "binance,bybit,okx,kucoin,gateio,mexc,bitget,coinex").split(",") if x.strip()]
EXCHANGE_UNIVERSE = int(os.getenv("EXCHANGE_UNIVERSE", "20"))
MULTI_MIN_TURNOVER = float(os.getenv("MULTI_MIN_TURNOVER_24H", "500000"))
EMA200_MIN_SCORE = int(os.getenv("EMA200_MIN_SCORE", "50"))

CATEGORY = {"SPOT": "SPOT", "FUTURES": "USDT-FUTURES"}
STABLE_BASES = {
    "USDT", "USDC", "FDUSD", "TUSD", "DAI", "USDE", "USDD", "PYUSD", "USD1", "USDS", "EURC"
}

@dataclass
class Signal:
    signal_id: str
    symbol: str
    market: str
    side: str
    timeframe: str
    entry_low: float
    entry_high: float
    stop: float
    tp1: float
    tp2: float
    tp3: float
    score: int
    reasons: list[str]
    created_at: datetime

@dataclass
class GrowthCandidate:
    candidate_id: str
    symbol: str
    market: str
    price: float
    score: int
    change24h: float
    turnover24h: float
    rsi: float
    atr_pct: float
    distance_breakout_pct: float
    reasons: list[str]
    created_at: datetime
    radar_type: str = "GROWTH"
    below_ema200_4h: bool = False
    below_ema200_1d: bool = False

pending: dict[str, Signal] = {}
pending_growth: dict[str, GrowthCandidate] = {}
paused = False
scan_lock = asyncio.Lock()
growth_lock = asyncio.Lock()
last_scan_at: Optional[datetime] = None
last_growth_at: Optional[datetime] = None


def is_admin(update: Update) -> bool:
    return bool(ADMIN_CHAT_ID) and str(update.effective_chat.id) == ADMIN_CHAT_ID


def fmt_price(x: float) -> str:
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    return f"{x:.8f}".rstrip("0").rstrip(".")


def get_json(path: str, params: dict) -> dict:
    url = f"{BASE_URL}{path}"
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if str(data.get("code")) not in ("00000", "0"):
        raise RuntimeError(f"Bitget error {data.get('code')}: {data.get('msg')}")
    return data


def candles(symbol: str, market: str, timeframe: str, limit: int = 250) -> pd.DataFrame:
    data = get_json("/api/v3/market/candles", {
        "category": CATEGORY[market],
        "symbol": symbol,
        "interval": timeframe,
        "limit": min(limit, 1000),
        "type": "market",
    })["data"]
    if not data:
        raise RuntimeError("empty candles")
    rows = [{
        "ts": int(r[0]), "open": float(r[1]), "high": float(r[2]),
        "low": float(r[3]), "close": float(r[4]), "volume": float(r[5]),
    } for r in data]
    df = pd.DataFrame(rows).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    if len(df) > 2:
        df = df.iloc[:-1].copy()
    return df


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False).mean()
    ad = down.ewm(alpha=1/n, adjust=False).mean()
    rs = au / ad.replace(0, float("nan"))
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def analyze(primary: pd.DataFrame, confirm: pd.DataFrame, market: str, symbol: str) -> Optional[Signal]:
    if len(primary) < 80 or len(confirm) < 80:
        return None
    p, c = primary.copy(), confirm.copy()
    for frame in (p, c):
        frame["ema20"] = ema(frame["close"], 20)
        frame["ema50"] = ema(frame["close"], 50)
        frame["ema200"] = ema(frame["close"], 200)
        frame["rsi"] = rsi(frame["close"], 14)
        frame["atr"] = atr(frame, 14)
        frame["vol_ma"] = frame["volume"].rolling(20).mean()

    row = p.iloc[-1]
    price, a = float(row.close), float(row.atr)
    if a <= 0:
        return None
    prev_high = float(p["high"].iloc[-BREAKOUT_LOOKBACK-1:-1].max())
    prev_low = float(p["low"].iloc[-BREAKOUT_LOOKBACK-1:-1].min())
    vol_ratio = float(row.volume / row.vol_ma) if row.vol_ma > 0 else 0
    long_htf = c.iloc[-1].close > c.iloc[-1].ema50 and c.iloc[-1].ema20 > c.iloc[-1].ema50
    short_htf = c.iloc[-1].close < c.iloc[-1].ema50 and c.iloc[-1].ema20 < c.iloc[-1].ema50
    long_score = short_score = 0
    long_reasons, short_reasons = [], []

    if row.close > row.ema20 > row.ema50 > row.ema200:
        long_score += 25; long_reasons.append("EMA20>EMA50>EMA200 trend")
    if row.close < row.ema20 < row.ema50 < row.ema200:
        short_score += 25; short_reasons.append("EMA20<EMA50<EMA200 trend")
    if 52 <= row.rsi <= 72:
        long_score += 15; long_reasons.append(f"RSI {row.rsi:.0f}")
    if 28 <= row.rsi <= 48:
        short_score += 15; short_reasons.append(f"RSI {row.rsi:.0f}")
    if row.close > prev_high and row.close > p.iloc[-2].close:
        long_score += 20; long_reasons.append(f"{BREAKOUT_LOOKBACK}-candle breakout")
    if row.close < prev_low and row.close < p.iloc[-2].close:
        short_score += 20; short_reasons.append(f"{BREAKOUT_LOOKBACK}-candle breakdown")
    if vol_ratio >= 1.25:
        long_score += 15; short_score += 15
        long_reasons.append(f"volume {vol_ratio:.1f}x"); short_reasons.append(f"volume {vol_ratio:.1f}x")
    if long_htf:
        long_score += 15; long_reasons.append(f"{CONFIRM_TF} trend confirms")
    if short_htf:
        short_score += 15; short_reasons.append(f"{CONFIRM_TF} trend confirms")
    if market == "SPOT":
        short_score = -1

    side = "LONG" if long_score >= short_score else "SHORT"
    score = long_score if side == "LONG" else short_score
    reasons = long_reasons if side == "LONG" else short_reasons
    core_text = " ".join(reasons).lower()
    core = "breakout" in core_text if side == "LONG" else "breakdown" in core_text
    if score < MIN_SCORE or not core:
        return None

    if side == "LONG":
        entry_low, entry_high = price - 0.20*a, price + 0.10*a
        stop = price - 1.20*a
        risk = price - stop
        tp1, tp2, tp3 = price + risk, price + 2*risk, price + 3*risk
    else:
        entry_low, entry_high = price - 0.10*a, price + 0.20*a
        stop = price + 1.20*a
        risk = stop - price
        tp1, tp2, tp3 = price - risk, price - 2*risk, price - 3*risk

    sid = f"{market}:{symbol}:{side}:{int(row.ts)}"
    return Signal(sid, symbol, market, side, PRIMARY_TF,
                  min(entry_low, entry_high), max(entry_low, entry_high),
                  stop, tp1, tp2, tp3, int(score), reasons,
                  datetime.now(timezone.utc))


async def fetch_analysis(symbol: str, market: str) -> Optional[Signal]:
    try:
        primary, confirm = await asyncio.gather(
            asyncio.to_thread(candles, symbol, market, PRIMARY_TF),
            asyncio.to_thread(candles, symbol, market, CONFIRM_TF),
        )
        return analyze(primary, confirm, market, symbol)
    except Exception as e:
        log.warning("signal scan failed %s %s: %s", market, symbol, e)
        return None


def ticker_rows(market: str) -> list[dict]:
    data = get_json("/api/v3/market/tickers", {"category": CATEGORY[market]})["data"]
    rows = []
    for x in data:
        symbol = str(x.get("symbol", "")).upper()
        if not symbol.endswith("USDT"):
            continue
        base = symbol[:-4]
        if base in STABLE_BASES:
            continue
        try:
            last = float(x.get("lastPrice", 0))
            turnover = float(x.get("turnover24h", 0))
            change = float(x.get("price24hPcnt", 0)) * 100
        except (TypeError, ValueError):
            continue
        if last <= 0 or turnover < GROWTH_MIN_TURNOVER:
            continue
        rows.append({"symbol": symbol, "price": last, "turnover": turnover, "change": change})
    return rows


def growth_metrics(df: pd.DataFrame) -> Optional[dict]:
    if len(df) < 80:
        return None
    d = df.copy()
    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    d["rsi"] = rsi(d["close"], 14)
    d["atr"] = atr(d, 14)
    d["vol_ma"] = d["volume"].rolling(20).mean()
    row = d.iloc[-1]
    price = float(row.close)
    if price <= 0 or row.atr <= 0:
        return None
    breakout = float(d["high"].iloc[-21:-1].max())
    distance = (breakout - price) / price * 100
    vol_ratio = float(row.volume / row.vol_ma) if row.vol_ma > 0 else 0
    return {
        "price": price,
        "ema20": float(row.ema20), "ema50": float(row.ema50), "ema200": float(row.ema200),
        "rsi": float(row.rsi), "atr_pct": float(row.atr / price * 100),
        "distance": distance, "vol_ratio": vol_ratio,
        "ema200_gap_pct": (price - float(row.ema200)) / float(row.ema200) * 100,
    }


def recovery_score(m1h: dict, m4h: dict, m1d: dict, ticker: dict) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    # The requested radar is specifically for coins trading below EMA200 on BOTH 4H and 1D.
    if m4h["price"] < m4h["ema200"]:
        score += 25
        reasons.append(f"4H below EMA200 ({m4h['ema200_gap_pct']:.1f}%)")
    if m1d["price"] < m1d["ema200"]:
        score += 25
        reasons.append(f"1D below EMA200 ({m1d['ema200_gap_pct']:.1f}%)")

    # Prefer names that are close enough to EMA200 to have a plausible recovery path.
    avg_gap = (abs(m4h["ema200_gap_pct"]) + abs(m1d["ema200_gap_pct"])) / 2
    if avg_gap <= 8:
        score += 15; reasons.append(f"close to EMA200 (avg {avg_gap:.1f}% below)")
    elif avg_gap <= 15:
        score += 10; reasons.append(f"{avg_gap:.1f}% below EMA200 on average")
    elif avg_gap <= 30:
        score += 5; reasons.append(f"{avg_gap:.1f}% below EMA200 on average")

    # Early-recovery structure on 4H.
    if m4h["ema20"] > m4h["ema50"]:
        score += 10; reasons.append("4H EMA20 > EMA50")
    if 40 <= m4h["rsi"] <= 60:
        score += 8; reasons.append(f"4H RSI {m4h['rsi']:.0f}")
    if 35 <= m1d["rsi"] <= 58:
        score += 7; reasons.append(f"1D RSI {m1d['rsi']:.0f}")

    vr = m1h["vol_ratio"]
    if vr >= 1.5:
        score += 5; reasons.append(f"1H volume {vr:.1f}x")
    elif vr >= 1.15:
        score += 3; reasons.append(f"1H volume {vr:.1f}x")

    ch = ticker["change"]
    if 0 <= ch <= 8:
        score += 5; reasons.append(f"24h change {ch:+.1f}%")
    elif -5 <= ch < 0:
        score += 2; reasons.append(f"24h change {ch:+.1f}%")
    return score, reasons


def build_growth_candidate(symbol: str, market: str, ticker: dict, metrics: dict) -> Optional[GrowthCandidate]:
    price = metrics["price"]
    score = 0
    reasons = []
    change = ticker["change"]

    if 2 <= change <= 12:
        score += 20; reasons.append(f"24h momentum +{change:.1f}%")
    elif 0 <= change < 2:
        score += 8; reasons.append(f"24h momentum +{change:.1f}%")
    elif 12 < change <= 20:
        score += 12; reasons.append(f"24h momentum +{change:.1f}% (extended)")
    elif change > 20:
        score += 3; reasons.append(f"24h +{change:.1f}% (highly extended)")
    else:
        return None

    if price > metrics["ema20"] > metrics["ema50"]:
        score += 15; reasons.append("price > EMA20 > EMA50")
    elif price > metrics["ema50"]:
        score += 8; reasons.append("price above EMA50")
    else:
        return None
    if metrics["ema50"] > metrics["ema200"]:
        score += 10; reasons.append("EMA50 above EMA200")

    r = metrics["rsi"]
    if 55 <= r <= 68:
        score += 20; reasons.append(f"RSI {r:.0f} constructive")
    elif 50 <= r < 55 or 68 < r <= 72:
        score += 12; reasons.append(f"RSI {r:.0f} acceptable")
    elif 72 < r <= 78:
        score += 5; reasons.append(f"RSI {r:.0f} hot")
    else:
        return None

    vr = metrics["vol_ratio"]
    if vr >= 1.5:
        score += 15; reasons.append(f"volume {vr:.1f}x")
    elif vr >= 1.15:
        score += 9; reasons.append(f"volume {vr:.1f}x")

    dist = metrics["distance"]
    if 0 <= dist <= 2.5:
        score += 20; reasons.append(f"within {dist:.1f}% of 20H breakout")
    elif 2.5 < dist <= 5:
        score += 12; reasons.append(f"{dist:.1f}% below 20H breakout")
    elif 5 < dist <= 8:
        score += 6; reasons.append(f"{dist:.1f}% below 20H breakout")

    if score < GROWTH_MIN_SCORE:
        return None
    cid = f"GROWTH:{market}:{symbol}:{int(ticker.get('ts', 0) or 0)}:{int(score)}"
    return GrowthCandidate(
        cid, symbol, market, price, int(score), change, ticker["turnover"],
        metrics["rsi"], metrics["atr_pct"], dist, reasons, datetime.now(timezone.utc),
        "GROWTH", False, False
    )


def build_ema200_candidate(symbol: str, market: str, ticker: dict, m1h: dict, m4h: dict, m1d: dict) -> Optional[GrowthCandidate]:
    if not (m4h["price"] < m4h["ema200"] and m1d["price"] < m1d["ema200"]):
        return None
    score, reasons = recovery_score(m1h, m4h, m1d, ticker)
    if score < EMA200_MIN_SCORE:
        return None
    # Use the 1H market price/ticker for the displayed live value.
    price = m1h["price"]
    dist = m1h["distance"]
    cid = f"EMA200:{market}:{symbol}:{int(ticker.get('ts', 0) or 0)}:{int(score)}"
    return GrowthCandidate(
        cid, symbol, market, price, int(score), ticker["change"], ticker["turnover"],
        m4h["rsi"], m1h["atr_pct"], dist, reasons, datetime.now(timezone.utc),
        "EMA200_RADAR", True, True
    )


async def _ccxt_exchange(exchange_id: str, market_type: str):
    """Create a CCXT exchange using public endpoints only."""
    cls = getattr(ccxt, exchange_id, None)
    if cls is None:
        raise RuntimeError(f"unsupported ccxt exchange: {exchange_id}")
    options = {"enableRateLimit": True, "timeout": REQUEST_TIMEOUT * 1000}
    if market_type == "FUTURES":
        options["options"] = {"defaultType": "swap"}
    return cls(options)


def _multi_markets(exchange_id: str, market_type: str):
    ex = _ccxt_exchange(exchange_id, market_type)
    markets = ex.load_markets()
    wanted = []
    for sym, m in markets.items():
        if not m.get("active", True):
            continue
        if m.get("quote") != "USDT":
            continue
        if market_type == "SPOT":
            if not m.get("spot"):
                continue
        else:
            if not (m.get("swap") or m.get("future")):
                continue
        base = str(m.get("base", "")).upper()
        if base in STABLE_BASES:
            continue
        wanted.append(sym)
    return ex, wanted


def _multi_tickers(exchange_id: str, market_type: str) -> list[dict]:
    ex, symbols = _multi_markets(exchange_id, market_type)
    try:
        raw = ex.fetch_tickers(symbols if len(symbols) <= 1000 else symbols[:1000])
    finally:
        try:
            ex.close()
        except Exception:
            pass
    rows = []
    for sym, t in raw.items():
        try:
            last = float(t.get("last") or 0)
            quote_vol = float(t.get("quoteVolume") or 0)
            change = float(t.get("percentage") or 0)
        except (TypeError, ValueError):
            continue
        if last <= 0 or quote_vol < MULTI_MIN_TURNOVER:
            continue
        rows.append({"symbol": sym, "display": sym, "price": last,
                     "turnover": quote_vol, "change": change,
                     "ts": int(t.get("timestamp") or 0)})
    return sorted(rows, key=lambda x: x["turnover"], reverse=True)[:EXCHANGE_UNIVERSE]


def _multi_candles(exchange_id: str, market_type: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ex = _ccxt_exchange(exchange_id, market_type)
    tf = {"1H": "1h", "4H": "4h", "1D": "1d"}.get(timeframe.upper(), timeframe.lower())
    try:
        rows = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    finally:
        try:
            ex.close()
        except Exception:
            pass
    if not rows:
        raise RuntimeError("empty candles")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    if len(df) > 2:
        df = df.iloc[:-1].copy()
    return df


async def _scan_exchange(exchange_id: str, market_type: str) -> list[GrowthCandidate]:
    out: list[GrowthCandidate] = []
    try:
        tickers = await asyncio.to_thread(_multi_tickers, exchange_id, market_type)
    except Exception as e:
        log.warning("exchange ticker scan failed %s %s: %s", exchange_id, market_type, e)
        return out
    for t in tickers:
        try:
            df1h, df4h, df1d = await asyncio.gather(
                asyncio.to_thread(_multi_candles, exchange_id, market_type, t["symbol"], "1H", 260),
                asyncio.to_thread(_multi_candles, exchange_id, market_type, t["symbol"], "4H", 260),
                asyncio.to_thread(_multi_candles, exchange_id, market_type, t["symbol"], "1D", 260),
            )
            m1h, m4h, m1d = growth_metrics(df1h), growth_metrics(df4h), growth_metrics(df1d)
            if not (m1h and m4h and m1d):
                continue
            display_symbol = f"{t['display']} @ {exchange_id.upper()}"
            t2 = dict(t); t2["change"] = t["change"]; t2["symbol"] = display_symbol
            c = build_growth_candidate(display_symbol, f"{exchange_id.upper()}:{market_type}", t2, m1h)
            if c:
                out.append(c)
            e = build_ema200_candidate(display_symbol, f"{exchange_id.upper()}:{market_type}", t2, m1h, m4h, m1d)
            if e:
                out.append(e)
        except Exception as e:
            log.debug("multi-exchange scan failed %s %s %s: %s", exchange_id, market_type, t["symbol"], e)
    return out


async def growth_scan() -> list[GrowthCandidate]:
    """Scan Growth + EMA200 across configured major exchanges.
    Falls back to the existing Bitget scanner if multi-exchange mode is disabled.
    """
    if MULTI_EXCHANGE_ENABLED:
        jobs = [
            _scan_exchange(ex, mt)
            for ex in EXCHANGES
            for mt in ("SPOT", "FUTURES")
        ]
        results = await asyncio.gather(*jobs, return_exceptions=True)
        candidates = []
        for r in results:
            if isinstance(r, list):
                candidates.extend(r)
        return sorted(candidates, key=lambda x: (x.score, x.change24h), reverse=True)[:GROWTH_RESULTS * 4]

    candidates: list[GrowthCandidate] = []
    for market in MARKETS:
        if market not in CATEGORY:
            continue
        try:
            tickers = await asyncio.to_thread(ticker_rows, market)
            tickers = sorted(tickers, key=lambda x: x["turnover"], reverse=True)[:GROWTH_UNIVERSE]
            for t in tickers:
                try:
                    df1h, df4h, df1d = await asyncio.gather(
                        asyncio.to_thread(candles, t["symbol"], market, PRIMARY_TF, 260),
                        asyncio.to_thread(candles, t["symbol"], market, "4H", 260),
                        asyncio.to_thread(candles, t["symbol"], market, "1D", 260),
                    )
                    m1h, m4h, m1d = growth_metrics(df1h), growth_metrics(df4h), growth_metrics(df1d)
                    if not (m1h and m4h and m1d):
                        continue
                    c = build_growth_candidate(t["symbol"], market, t, m1h)
                    if c: candidates.append(c)
                    e = build_ema200_candidate(t["symbol"], market, t, m1h, m4h, m1d)
                    if e: candidates.append(e)
                except Exception as e:
                    log.debug("growth failed %s %s: %s", market, t["symbol"], e)
        except Exception as e:
            log.warning("growth universe failed %s: %s", market, e)
    return sorted(candidates, key=lambda x: (x.score, x.change24h), reverse=True)


def growth_text(c: GrowthCandidate, pending_mode=True) -> str:
    status = "⏳ در انتظار تأیید" if pending_mode else "📢 منتشر شد"
    reasons = "\n".join(f"• {x}" for x in c.reasons)
    title = "📡 <b>EMA200 RADAR V2</b>" if c.radar_type == "EMA200_RADAR" else "🚀 <b>GROWTH WATCHLIST V2</b>"
    subtitle = (
        "ارز زیر EMA200 در 4H و 1D؛ این بخش برای پیدا کردن کاندیدای احتمالی برگشت ساخته شده است."
        if c.radar_type == "EMA200_RADAR" else
        "رتبه‌بندی تکنیکال ارزهای مستعد رشد."
    )
    return (
        f"{title}\n\n"
        f"<b>{c.symbol}</b> | {c.market}\n"
        f"Price: <b>{fmt_price(c.price)}</b>\n"
        f"Score: <b>{c.score}/100</b>\n"
        f"24h: <b>{c.change24h:+.2f}%</b> | RSI 4H: <b>{c.rsi:.0f}</b>\n"
        f"ATR 1H: <b>{c.atr_pct:.2f}%</b> | 20H breakout distance: <b>{c.distance_breakout_pct:.2f}%</b>\n"
        f"24h turnover: <b>${c.turnover24h:,.0f}</b>\n\n"
        f"<b>دلایل:</b>\n{reasons}\n\n"
        f"{status}\n"
        f"<i>{subtitle} تضمین رشد یا برگشت نیست و صرفاً بر اساس داده‌های بازار است.</i>"
    )


async def run_scan(app: Application, manual=False):
    global last_scan_at
    if scan_lock.locked():
        return
    async with scan_lock:
        last_scan_at = datetime.now(timezone.utc)
        jobs = [
            fetch_analysis(symbol, market)
            for market in MARKETS if market in CATEGORY
            for symbol in SYMBOLS
        ]
        results = await asyncio.gather(*jobs)
        found = sorted((x for x in results if x), key=lambda x: x.score, reverse=True)
        for s in found[:MAX_PENDING]:
            if s.signal_id in pending:
                continue
            pending[s.signal_id] = s
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ APPROVE", callback_data=f"approve:{s.signal_id}"),
                 InlineKeyboardButton("❌ REJECT", callback_data=f"reject:{s.signal_id}")],
                [InlineKeyboardButton("🔄 RECHECK", callback_data=f"recheck:{s.signal_id}")]
            ])
            await app.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text=signal_text(s, True),
                                       parse_mode=ParseMode.HTML, reply_markup=kb)
        if manual and not found:
            await app.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text="🔎 اسکن انجام شد؛ فعلاً سیگنال واجد شرایط پیدا نشد.")


async def run_growth(app: Application, manual=False):
    global last_growth_at
    if growth_lock.locked():
        return
    async with growth_lock:
        last_growth_at = datetime.now(timezone.utc)
        found = await growth_scan()
        for c in found[:GROWTH_RESULTS]:
            if c.candidate_id in pending_growth:
                continue
            pending_growth[c.candidate_id] = c
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("📢 APPROVE WATCHLIST", callback_data=f"gapprove:{c.candidate_id}"),
                 InlineKeyboardButton("❌ REJECT", callback_data=f"greject:{c.candidate_id}")],
                [InlineKeyboardButton("🔄 RECHECK", callback_data=f"grecheck:{c.candidate_id}")]
            ])
            await app.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text=growth_text(c, True),
                                       parse_mode=ParseMode.HTML, reply_markup=kb)
        if manual and not found:
            await app.bot.send_message(chat_id=int(ADMIN_CHAT_ID),
                                       text="🚀 اسکن رشد انجام شد؛ فعلاً کاندیدای رشد با امتیاز لازم پیدا نشد.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚀 <b>Hiwacrypto Bot V2</b> آنلاین است.\n\n"
        "/scan — اسکن سیگنال\n"
        "/growth — پیدا کردن ارزهای مستعد رشد\n"
        "/status — وضعیت تنظیمات\n"
        "/pause — توقف اسکن خودکار\n"
        "/resume — ادامه اسکن خودکار",
        parse_mode=ParseMode.HTML,
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    await update.message.reply_text(
        f"🟢 V2 ONLINE\n"
        f"Markets: {', '.join(MARKETS)}\n"
        f"Symbols: {len(SYMBOLS)}\n"
        f"Primary TF: {PRIMARY_TF}\n"
        f"Confirm TF: {CONFIRM_TF}\n"
        f"Min score: {MIN_SCORE}\n"
        f"Signal interval: {SCAN_INTERVAL_MIN}m\n"
        f"Growth scanner: {'ON' if GROWTH_ENABLED else 'OFF'}\n"
        f"Growth interval: {GROWTH_INTERVAL_MIN}m\n"
        f"Growth universe: {GROWTH_UNIVERSE}/market\n"
        f"Growth min score: {GROWTH_MIN_SCORE}\n"
        f"EMA200 radar: 4H + 1D below EMA200 | min score {EMA200_MIN_SCORE}\n"
        f"Pending signals: {len(pending)}\n"
        f"Pending growth: {len(pending_growth)}\n"
        f"Paused: {paused}"
    )


async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    await update.message.reply_text("🔎 در حال اسکن Spot + Futures ...")
    await run_scan(context.application, manual=True)


async def growth_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    await update.message.reply_text("🚀 در حال پیدا کردن ارزهای مستعد رشد در Bitget ...")
    await run_growth(context.application, manual=True)


async def pause_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global paused
    if not is_admin(update):
        return
    paused = True
    await update.message.reply_text("⏸ اسکن‌های خودکار متوقف شد.")


async def resume_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global paused
    if not is_admin(update):
        return
    paused = False
    await update.message.reply_text("▶️ اسکن‌های خودکار ادامه پیدا کرد.")


async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q or not q.message or not is_admin(update):
        return
    await q.answer()
    action, sid = q.data.split(":", 1)

    if action in {"gapprove", "greject", "grecheck"}:
        c = pending_growth.get(sid)
        if not c:
            await q.edit_message_text("این کاندیدای رشد منقضی شده یا قبلاً پردازش شده.")
            return
        if action == "gapprove":
            await context.bot.send_message(chat_id=CHANNEL, text=growth_text(c, False), parse_mode=ParseMode.HTML)
            pending_growth.pop(sid, None)
            await q.edit_message_text("📢 واچ‌لیست رشد تأیید و به کانال ارسال شد.")
        elif action == "greject":
            pending_growth.pop(sid, None)
            await q.edit_message_text("❌ کاندیدای رشد رد شد.")
        else:
            pending_growth.pop(sid, None)
            await q.edit_message_text("🔄 بررسی مجدد کاندیدای رشد...")
            await run_growth(context.application, manual=False)
        return

    s = pending.get(sid)
    if not s:
        await q.edit_message_text("این سیگنال منقضی شده یا قبلاً پردازش شده.")
        return
    if action == "approve":
        await context.bot.send_message(chat_id=CHANNEL, text=signal_text(s, False), parse_mode=ParseMode.HTML)
        pending.pop(sid, None)
        await q.edit_message_text("✅ سیگنال تأیید شد و به کانال ارسال شد.")
    elif action == "reject":
        pending.pop(sid, None)
        await q.edit_message_text("❌ سیگنال رد شد.")
    elif action == "recheck":
        pending.pop(sid, None)
        await q.edit_message_text("🔄 در حال بررسی مجدد...")
        ns = await fetch_analysis(s.symbol, s.market)
        if ns:
            pending[ns.signal_id] = ns
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ APPROVE", callback_data=f"approve:{ns.signal_id}"),
                 InlineKeyboardButton("❌ REJECT", callback_data=f"reject:{ns.signal_id}")],
                [InlineKeyboardButton("🔄 RECHECK", callback_data=f"recheck:{ns.signal_id}")]
            ])
            await context.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text=signal_text(ns, True),
                                           parse_mode=ParseMode.HTML, reply_markup=kb)
        else:
            await context.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text="🔄 در بررسی مجدد، سیگنال دیگر شرایط لازم را نداشت.")


async def background_signal_scanner(app: Application):
    await asyncio.sleep(10)
    while True:
        try:
            if not paused:
                await run_scan(app)
        except Exception:
            log.exception("signal scanner error")
        await asyncio.sleep(max(1, SCAN_INTERVAL_MIN) * 60)


async def background_growth_scanner(app: Application):
    await asyncio.sleep(20)
    while True:
        try:
            if not paused and GROWTH_ENABLED:
                await run_growth(app)
        except Exception:
            log.exception("growth scanner error")
        await asyncio.sleep(max(1, GROWTH_INTERVAL_MIN) * 60)


async def post_init(app: Application):
    asyncio.create_task(background_signal_scanner(app))
    if GROWTH_ENABLED:
        asyncio.create_task(background_growth_scanner(app))
    await app.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text="🚀 Hiwacrypto V2 + Growth Scanner آنلاین شد. اسکن خودکار فعال است.")


def main():
    if not TOKEN or not ADMIN_CHAT_ID:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and ADMIN_CHAT_ID in Railway Variables.")
    application = Application.builder().token(TOKEN).post_init(post_init).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("scan", scan_cmd))
    application.add_handler(CommandHandler("growth", growth_cmd))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("pause", pause_cmd))
    application.add_handler(CommandHandler("resume", resume_cmd))
    application.add_handler(CallbackQueryHandler(callback))
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
