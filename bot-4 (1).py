import os
import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests
import pandas as pd
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

CATEGORY = {"SPOT": "SPOT", "FUTURES": "USDT-FUTURES"}

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

pending: dict[str, Signal] = {}
paused = False
scan_lock = asyncio.Lock()
last_scan_at: Optional[datetime] = None

def is_admin(update: Update) -> bool:
    return bool(ADMIN_CHAT_ID) and str(update.effective_chat.id) == ADMIN_CHAT_ID

def fmt_price(x: float) -> str:
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    return f"{x:.8f}".rstrip("0").rstrip(".")

BITGET_REQUEST_GAP = float(os.getenv("BITGET_REQUEST_GAP_SECONDS", "0.35"))
BITGET_MAX_RETRIES = int(os.getenv("BITGET_MAX_RETRIES", "4"))
_bitget_request_lock = threading.Lock()
_bitget_last_request_at = 0.0

def get_json(path: str, params: dict) -> dict:
    global _bitget_last_request_at
    url = f"{BASE_URL}{path}"
    for attempt in range(BITGET_MAX_RETRIES + 1):
        with _bitget_request_lock:
            now = time.monotonic()
            wait = BITGET_REQUEST_GAP - (now - _bitget_last_request_at)
            if wait > 0:
                time.sleep(wait)
            _bitget_last_request_at = time.monotonic()
            try:
                r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            except requests.RequestException:
                if attempt >= BITGET_MAX_RETRIES:
                    raise
                time.sleep(min(2 ** attempt, 8))
                continue
        if r.status_code == 429:
            if attempt >= BITGET_MAX_RETRIES:
                r.raise_for_status()
            retry_after = r.headers.get("Retry-After", "")
            try:
                server_wait = float(retry_after)
            except (TypeError, ValueError):
                server_wait = 0.0
            backoff = max(server_wait, 1.5 * (2 ** attempt))
            log.warning("Bitget rate limit (429) %s; retrying in %.1fs [%d/%d]", path, backoff, attempt + 1, BITGET_MAX_RETRIES)
            time.sleep(min(backoff, 15))
            continue
        r.raise_for_status()
        data = r.json()
        if str(data.get("code")) not in ("00000", "0"):
            raise RuntimeError(f"Bitget error {data.get('code')}: {data.get('msg')}")
        return data
    raise RuntimeError("Bitget request failed after retries")

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
        "ts": int(r[0]),
        "open": float(r[1]),
        "high": float(r[2]),
        "low": float(r[3]),
        "close": float(r[4]),
        "volume": float(r[5]),
    } for r in data]
    df = pd.DataFrame(rows).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    # Ignore the newest candle because it may still be forming.
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

    p = primary.copy()
    c = confirm.copy()
    for frame in (p, c):
        frame["ema20"] = ema(frame["close"], 20)
        frame["ema50"] = ema(frame["close"], 50)
        frame["ema200"] = ema(frame["close"], 200)
        frame["rsi"] = rsi(frame["close"], 14)
        frame["atr"] = atr(frame, 14)
        frame["vol_ma"] = frame["volume"].rolling(20).mean()

    row = p.iloc[-1]
    price = float(row.close)
    a = float(row.atr)
    if a <= 0:
        return None

    prev_high = float(p["high"].iloc[-BREAKOUT_LOOKBACK-1:-1].max())
    prev_low = float(p["low"].iloc[-BREAKOUT_LOOKBACK-1:-1].min())
    vol_ratio = float(row.volume / row.vol_ma) if row.vol_ma > 0 else 0
    long_htf = c.iloc[-1].close > c.iloc[-1].ema50 and c.iloc[-1].ema20 > c.iloc[-1].ema50
    short_htf = c.iloc[-1].close < c.iloc[-1].ema50 and c.iloc[-1].ema20 < c.iloc[-1].ema50

    long_score = 0
    short_score = 0
    long_reasons, short_reasons = [], []

    if row.close > row.ema20 > row.ema50 > row.ema200:
        long_score += 25
        long_reasons.append("EMA20>EMA50>EMA200 trend")
    if row.close < row.ema20 < row.ema50 < row.ema200:
        short_score += 25
        short_reasons.append("EMA20<EMA50<EMA200 trend")

    if 52 <= row.rsi <= 72:
        long_score += 15
        long_reasons.append(f"RSI {row.rsi:.0f}")
    if 28 <= row.rsi <= 48:
        short_score += 15
        short_reasons.append(f"RSI {row.rsi:.0f}")

    if row.close > prev_high and row.close > p.iloc[-2].close:
        long_score += 20
        long_reasons.append(f"{BREAKOUT_LOOKBACK}-candle breakout")
    if row.close < prev_low and row.close < p.iloc[-2].close:
        short_score += 20
        short_reasons.append(f"{BREAKOUT_LOOKBACK}-candle breakdown")

    if vol_ratio >= 1.25:
        long_score += 15
        short_score += 15
        long_reasons.append(f"volume {vol_ratio:.1f}x")
        short_reasons.append(f"volume {vol_ratio:.1f}x")

    if long_htf:
        long_score += 15
        long_reasons.append(f"{CONFIRM_TF} trend confirms")
    if short_htf:
        short_score += 15
        short_reasons.append(f"{CONFIRM_TF} trend confirms")

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
    return Signal(
        sid, symbol, market, side, PRIMARY_TF,
        min(entry_low, entry_high), max(entry_low, entry_high),
        stop, tp1, tp2, tp3, int(score), reasons,
        datetime.now(timezone.utc)
    )

async def fetch_analysis(symbol: str, market: str) -> Optional[Signal]:
    try:
        primary, confirm = await asyncio.gather(
            asyncio.to_thread(candles, symbol, market, PRIMARY_TF),
            asyncio.to_thread(candles, symbol, market, CONFIRM_TF),
        )
        return analyze(primary, confirm, market, symbol)
    except Exception as e:
        log.warning("scan failed %s %s: %s", market, symbol, e)
        return None

def signal_text(s: Signal, pending_mode=True) -> str:
    icon = "🟢" if s.side == "LONG" else "🔴"
    status = "⏳ در انتظار تأیید" if pending_mode else "📢 منتشر شد"
    reasons = "\n".join(f"• {x}" for x in s.reasons)
    return (
        f"{icon} <b>HIWACRYPTO V2 SIGNAL</b>\n\n"
        f"<b>{s.symbol}</b> | {s.market} | <b>{s.side}</b>\n"
        f"TF: <b>{s.timeframe}</b> | Score: <b>{s.score}/100</b>\n"
        f"Entry: <b>{fmt_price(s.entry_low)} – {fmt_price(s.entry_high)}</b>\n"
        f"SL: <b>{fmt_price(s.stop)}</b>\n"
        f"TP1: <b>{fmt_price(s.tp1)}</b>\n"
        f"TP2: <b>{fmt_price(s.tp2)}</b>\n"
        f"TP3: <b>{fmt_price(s.tp3)}</b>\n\n"
        f"<b>دلایل:</b>\n{reasons}\n\n"
        f"{status}\n"
        f"<i>تحلیل خودکار؛ بدون اجرای معامله</i>"
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
                [
                    InlineKeyboardButton("✅ APPROVE", callback_data=f"approve:{s.signal_id}"),
                    InlineKeyboardButton("❌ REJECT", callback_data=f"reject:{s.signal_id}"),
                ],
                [InlineKeyboardButton("🔄 RECHECK", callback_data=f"recheck:{s.signal_id}")]
            ])
            await app.bot.send_message(
                chat_id=int(ADMIN_CHAT_ID),
                text=signal_text(s, True),
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )

        if manual and not found:
            await app.bot.send_message(
                chat_id=int(ADMIN_CHAT_ID),
                text="🔎 اسکن انجام شد؛ فعلاً سیگنال واجد شرایط پیدا نشد."
            )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚀 <b>Hiwacrypto Bot V2</b> آنلاین است.\n\n"
        "/scan — اسکن فوری\n"
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
        f"Interval: {SCAN_INTERVAL_MIN}m\n"
        f"Pending: {len(pending)}\n"
        f"Paused: {paused}"
    )

async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    await update.message.reply_text("🔎 در حال اسکن Spot + Futures ...")
    await run_scan(context.application, manual=True)

async def pause_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global paused
    if not is_admin(update):
        return
    paused = True
    await update.message.reply_text("⏸ اسکن خودکار متوقف شد.")

async def resume_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global paused
    if not is_admin(update):
        return
    paused = False
    await update.message.reply_text("▶️ اسکن خودکار ادامه پیدا کرد.")

async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q or not q.message or not is_admin(update):
        return
    await q.answer()
    action, sid = q.data.split(":", 1)
    s = pending.get(sid)
    if not s:
        await q.edit_message_text("این سیگنال منقضی شده یا قبلاً پردازش شده.")
        return

    if action == "approve":
        await context.bot.send_message(
            chat_id=CHANNEL,
            text=signal_text(s, False),
            parse_mode=ParseMode.HTML,
        )
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
                [
                    InlineKeyboardButton("✅ APPROVE", callback_data=f"approve:{ns.signal_id}"),
                    InlineKeyboardButton("❌ REJECT", callback_data=f"reject:{ns.signal_id}"),
                ],
                [InlineKeyboardButton("🔄 RECHECK", callback_data=f"recheck:{ns.signal_id}")]
            ])
            await context.bot.send_message(
                chat_id=int(ADMIN_CHAT_ID),
                text=signal_text(ns, True),
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )
        else:
            await context.bot.send_message(
                chat_id=int(ADMIN_CHAT_ID),
                text="🔄 در بررسی مجدد، سیگنال دیگر شرایط لازم را نداشت."
            )

async def background_scanner(app: Application):
    await asyncio.sleep(10)
    while True:
        try:
            if not paused:
                await run_scan(app)
        except Exception:
            log.exception("background scanner error")
        await asyncio.sleep(max(1, SCAN_INTERVAL_MIN) * 60)

async def post_init(app: Application):
    asyncio.create_task(background_scanner(app))
    await app.bot.send_message(
        chat_id=int(ADMIN_CHAT_ID),
        text="🚀 Hiwacrypto V2 آنلاین شد. اسکن خودکار فعال است."
    )

def main():
    if not TOKEN or not ADMIN_CHAT_ID:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and ADMIN_CHAT_ID in Railway Variables.")

    application = Application.builder().token(TOKEN).post_init(post_init).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("scan", scan_cmd))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("pause", pause_cmd))
    application.add_handler(CommandHandler("resume", resume_cmd))
    application.add_handler(CallbackQueryHandler(callback))
    application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
