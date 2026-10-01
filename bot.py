import os
import uuid
import math
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "").strip()
CHANNEL = os.getenv("CHANNEL_USERNAME", "@hiwacrypto").strip()
BASE_URL = os.getenv("BITGET_BASE_URL", "https://api.bitget.com").rstrip("/")
SYMBOLS = [x.strip().upper() for x in os.getenv(
    "SCAN_SYMBOLS",
    "BTCUSDT,SOLUSDT,ETHUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT"
).split(",") if x.strip()]
TIMEFRAME = os.getenv("TIMEFRAME", "4H").upper()

pending = {}

def api_get(path, params=None):
    r = requests.get(BASE_URL + path, params=params or {}, timeout=12)
    r.raise_for_status()
    data = r.json()
    if data.get("code") not in (None, "00000", 0):
        raise RuntimeError(str(data))
    return data.get("data", [])

def get_candles(symbol, granularity="4H", limit=120):
    # Bitget market candles endpoint. Public data only.
    rows = api_get(
        "/api/v2/mix/market/candles",
        {"symbol": symbol, "productType": "USDT-FUTURES",
         "granularity": granularity, "limit": str(limit)}
    )
    if not rows:
        raise RuntimeError("No candle data returned")
    # Typical Bitget order: timestamp, open, high, low, close, volume, quoteVolume
    df = pd.DataFrame(rows)
    df = df.iloc[:, :6]
    df.columns = ["ts","open","high","low","close","volume"]
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    return df.dropna().sort_values("ts").reset_index(drop=True)

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def build_candidate(symbol):
    df = get_candles(symbol, TIMEFRAME, 120)
    if len(df) < 60:
        return None

    close = df["close"]
    e20, e50 = ema(close, 20), ema(close, 50)
    last = float(close.iloc[-1])
    prev_high = float(df["high"].iloc[-21:-1].max())
    avg_vol = float(df["volume"].iloc[-21:-1].mean())
    vol = float(df["volume"].iloc[-1])

    # Simple v1 setup: trend + breakout + volume confirmation.
    long_setup = last > float(e20.iloc[-1]) > float(e50.iloc[-1]) and last > prev_high and vol > avg_vol * 1.20
    if not long_setup:
        return None

    entry_low = last * 0.997
    entry_high = last * 1.003
    sl = last * 0.975
    risk = last - sl

    return {
        "id": uuid.uuid4().hex[:8],
        "symbol": symbol,
        "side": "LONG",
        "market": "FUTURES",
        "timeframe": TIMEFRAME,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "sl": sl,
        "tp1": last + risk * 1.0,
        "tp2": last + risk * 2.0,
        "tp3": last + risk * 3.0,
        "reason": [
            "EMA20 > EMA50",
            "Price above recent 20-candle high",
            "Volume > 1.2× recent average"
        ]
    }

def fmt(x):
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.8f}"

def signal_text(s, pending=True):
    status = "🟡 PENDING APPROVAL" if pending else "🟢 APPROVED / PUBLISHED"
    reasons = "\n".join("• " + x for x in s["reason"])
    return (
        f"🚨 SIGNAL CANDIDATE\n\n"
        f"🪙 #{s['symbol']}\n"
        f"📍 {s['market']} — {s['side']}\n\n"
        f"Entry: {fmt(s['entry_low'])} – {fmt(s['entry_high'])}\n"
        f"SL: {fmt(s['sl'])}\n\n"
        f"TP1: {fmt(s['tp1'])}\n"
        f"TP2: {fmt(s['tp2'])}\n"
        f"TP3: {fmt(s['tp3'])}\n\n"
        f"⏱ TF: {s['timeframe']}\n\n"
        f"📊 Reasons:\n{reasons}\n\n"
        f"Status: {status}\n\n"
        f"⚠️ Analysis candidate — not a guarantee of profit."
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != ADMIN_CHAT_ID:
        await update.message.reply_text("Access restricted.")
        return
    await update.message.reply_text(
        "Hiwacrypto Bot v1 is online.\n\n"
        "/scan — scan the configured symbols\n"
        "/status — show configuration"
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != ADMIN_CHAT_ID:
        return
    await update.message.reply_text(
        f"Channel: {CHANNEL}\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Symbols: {', '.join(SYMBOLS)}"
    )

async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != ADMIN_CHAT_ID:
        return
    await update.message.reply_text("🔎 Scanning Bitget public market data...")
    found = 0
    for symbol in SYMBOLS:
        try:
            s = build_candidate(symbol)
            if not s:
                continue
            pending[s["id"]] = s
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ APPROVE", callback_data=f"approve:{s['id']}"),
                InlineKeyboardButton("❌ REJECT", callback_data=f"reject:{s['id']}")
            ],[
                InlineKeyboardButton("🔄 RECHECK", callback_data=f"recheck:{s['id']}")
            ]])
            await update.message.reply_text(signal_text(s), reply_markup=keyboard)
            found += 1
        except Exception as e:
            print(f"{symbol}: {e}")
    if found == 0:
        await update.message.reply_text("No v1 setups passed the current filters.")

async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if str(q.message.chat.id) != ADMIN_CHAT_ID:
        return

    action, sid = q.data.split(":", 1)
    s = pending.get(sid)
    if not s:
        await q.edit_message_text("Signal expired or already handled.")
        return

    if action == "reject":
        pending.pop(sid, None)
        await q.edit_message_text(signal_text(s, pending=False) + "\n\n❌ REJECTED")
        return

    if action == "recheck":
        try:
            fresh = build_candidate(s["symbol"])
            if not fresh:
                pending.pop(sid, None)
                await q.edit_message_text("🔄 Recheck: setup is no longer valid.")
                return
            fresh["id"] = sid
            pending[sid] = fresh
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ APPROVE", callback_data=f"approve:{sid}"),
                InlineKeyboardButton("❌ REJECT", callback_data=f"reject:{sid}")
            ],[
                InlineKeyboardButton("🔄 RECHECK", callback_data=f"recheck:{sid}")
            ]])
            await q.edit_message_text(signal_text(fresh), reply_markup=keyboard)
        except Exception as e:
            await q.edit_message_text(f"Recheck failed: {e}")
        return

    if action == "approve":
        try:
            await context.bot.send_message(
                chat_id=CHANNEL,
                text=signal_text(s, pending=False)
            )
            pending.pop(sid, None)
            await q.edit_message_text(signal_text(s, pending=False) + "\n\n📢 Published to " + CHANNEL)
        except Exception as e:
            await q.edit_message_text(
                signal_text(s) + f"\n\n⚠️ Publication failed: {e}\n"
                "Check that the bot is an admin of the channel with Post Messages permission."
            )

def main():
    if not TOKEN or not ADMIN_CHAT_ID:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and ADMIN_CHAT_ID in .env")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scan", scan))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CallbackQueryHandler(callback))
    print("Hiwacrypto Bot v1 running...")
    app.run_polling()

if __name__ == "__main__":
    main()
