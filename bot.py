import os, uuid, logging, requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
TOKEN=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
ADMIN_CHAT_ID=os.getenv("ADMIN_CHAT_ID","").strip()
CHANNEL=os.getenv("CHANNEL_USERNAME","@hiwacrypto").strip()
BASE=os.getenv("BITGET_BASE_URL","https://api.bitget.com").rstrip("/")
INTERVAL=int(os.getenv("SCAN_INTERVAL_SECONDS","900"))
MIN_SCORE=float(os.getenv("MIN_SCORE","70"))
TIMEFRAMES=[x.strip().upper() for x in os.getenv("TIMEFRAMES","1H,4H").split(",") if x.strip()]
SYMBOLS=[x.strip().upper() for x in os.getenv("SYMBOLS","BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT").split(",") if x.strip()]
pending={}; paused=False; last_scan="Not yet"

def api_get(path, params=None):
    r=requests.get(BASE+path, params=params or {}, timeout=15); r.raise_for_status()
    d=r.json()
    if d.get("code") not in (None,"00000",0): raise RuntimeError(str(d))
    return d.get("data",[])

def candles(symbol, market, tf, limit=150):
    path="/api/v2/spot/market/candles" if market=="SPOT" else "/api/v2/mix/market/candles"
    p={"symbol":symbol,"granularity":tf,"limit":str(limit)}
    if market!="SPOT": p["productType"]="USDT-FUTURES"
    rows=api_get(path,p)
    if not rows: raise RuntimeError("No candle data")
    df=pd.DataFrame(rows).iloc[:,:6]; df.columns=["ts","open","high","low","close","volume"]
    for c in ["open","high","low","close","volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.dropna().sort_values("ts").reset_index(drop=True)

def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False).mean(); ad=dn.ewm(alpha=1/n,adjust=False).mean()
    rs=au/ad.replace(0,np.nan); return 100-(100/(1+rs))

def evaluate(df):
    if len(df)<80: return None
    c=df.close; e20=ema(c,20); e50=ema(c,50); rv=float(rsi(c).iloc[-1])
    last=float(c.iloc[-1]); hi=float(df.high.iloc[-21:-1].max()); lo=float(df.low.iloc[-21:-1].min())
    av=float(df.volume.iloc[-21:-1].mean()); vol=float(df.volume.iloc[-1])
    L=S=0; lr=[]; sr=[]
    if e20.iloc[-1]>e50.iloc[-1]: L+=25; lr.append("EMA20 > EMA50")
    if e20.iloc[-1]<e50.iloc[-1]: S+=25; sr.append("EMA20 < EMA50")
    if last>hi: L+=30; lr.append("20-candle breakout")
    if last<lo: S+=30; sr.append("20-candle breakdown")
    if vol>av*1.2:
        if last>=float(c.iloc[-2]): L+=20; lr.append("Volume expansion")
        else: S+=20; sr.append("Volume expansion")
    if 50<=rv<=68: L+=15; lr.append(f"RSI supportive ({rv:.1f})")
    if 32<=rv<=50: S+=15; sr.append(f"RSI supportive ({rv:.1f})")
    if rv>75: L-=15
    if rv<25: S-=15
    if L>=S and L>=MIN_SCORE: side,score,reasons="LONG",L,lr
    elif S>L and S>=MIN_SCORE: side,score,reasons="SHORT",S,sr
    else: return None
    risk=last*0.025
    sl=last-risk if side=="LONG" else last+risk
    tps=[last+risk*i for i in (1,2,3)] if side=="LONG" else [last-risk*i for i in (1,2,3)]
    return side,score,last,sl,tps,reasons

def candidate(symbol,market,tf):
    x=evaluate(candles(symbol,market,tf))
    if not x:return None
    side,score,last,sl,tps,reasons=x
    return {"id":uuid.uuid4().hex[:8],"symbol":symbol,"market":market,"side":side,"timeframe":tf,
            "score":score,"entry_low":last*.997,"entry_high":last*1.003,"sl":sl,
            "tp1":tps[0],"tp2":tps[1],"tp3":tps[2],"reasons":reasons}

def fmt(x):
    return f"{x:.2f}" if x>=100 else (f"{x:.4f}" if x>=1 else f"{x:.8f}")

def render(s,status="🟡 PENDING APPROVAL"):
    reasons="\n".join("• "+x for x in s["reasons"])
    return (f"🚨 SIGNAL CANDIDATE\n\n🪙 #{s['symbol']}\n📍 {s['market']} — {s['side']}\n"
            f"⭐ Score: {s['score']:.0f}/100\n\nEntry: {fmt(s['entry_low'])} – {fmt(s['entry_high'])}\n"
            f"SL: {fmt(s['sl'])}\n\nTP1: {fmt(s['tp1'])}\nTP2: {fmt(s['tp2'])}\nTP3: {fmt(s['tp3'])}\n"
            f"\n⏱ TF: {s['timeframe']}\n\n📊 Reasons:\n{reasons}\n\nStatus: {status}\n\n"
            "⚠️ Candidate setup, not a profit guarantee.")

async def send_candidate(bot,s):
    pending[s["id"]]=s
    kb=InlineKeyboardMarkup([[InlineKeyboardButton("✅ APPROVE",callback_data=f"approve:{s['id']}"),
                              InlineKeyboardButton("❌ REJECT",callback_data=f"reject:{s['id']}")],
                             [InlineKeyboardButton("🔄 RECHECK",callback_data=f"recheck:{s['id']}")]])
    await bot.send_message(chat_id=ADMIN_CHAT_ID,text=render(s),reply_markup=kb)

async def do_scan(bot):
    global last_scan
    last_scan=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"); found=0; seen=set()
    for symbol in SYMBOLS:
        for market in ("SPOT","FUTURES"):
            for tf in TIMEFRAMES:
                try:
                    s=candidate(symbol,market,tf)
                    if not s: continue
                    key=(s["symbol"],s["market"],s["side"],s["timeframe"])
                    if key in seen: continue
                    seen.add(key); await send_candidate(bot,s); found+=1
                except Exception as e: logging.warning("%s %s %s: %s",symbol,market,tf,e)
    logging.info("Scan finished; %s candidates",found)

async def scheduled(context):
    if not paused: await do_scan(context.bot)

async def start(u:Update,c:ContextTypes.DEFAULT_TYPE):
    if str(u.effective_chat.id)!=ADMIN_CHAT_ID:return
    await u.message.reply_text("Hiwacrypto v2 online.\n/scan /status /pause /resume")

async def status(u,c):
    if str(u.effective_chat.id)!=ADMIN_CHAT_ID:return
    await u.message.reply_text(f"Channel: {CHANNEL}\nInterval: {INTERVAL}s\nTimeframes: {', '.join(TIMEFRAMES)}\n"
                               f"Minimum score: {MIN_SCORE}\nSymbols: {len(SYMBOLS)}\n"
                               f"Auto scan: {'PAUSED' if paused else 'ACTIVE'}\nLast scan: {last_scan}")

async def scan(u,c):
    if str(u.effective_chat.id)!=ADMIN_CHAT_ID:return
    await u.message.reply_text("🔎 Scanning Bitget...")
    await do_scan(c.bot); await u.message.reply_text("✅ Scan finished.")

async def pause(u,c):
    global paused
    if str(u.effective_chat.id)!=ADMIN_CHAT_ID:return
    paused=True; await u.message.reply_text("⏸ Automatic scanning paused.")

async def resume(u,c):
    global paused
    if str(u.effective_chat.id)!=ADMIN_CHAT_ID:return
    paused=False; await u.message.reply_text("▶️ Automatic scanning resumed.")

async def callback(u,c):
    q=u.callback_query; await q.answer()
    if str(q.message.chat.id)!=ADMIN_CHAT_ID:return
    action,sid=q.data.split(":",1); s=pending.get(sid)
    if not s: await q.edit_message_text("Signal expired or already handled."); return
    if action=="reject":
        pending.pop(sid,None); await q.edit_message_text(render(s,"❌ REJECTED")); return
    if action=="recheck":
        try:
            fresh=candidate(s["symbol"],s["market"],s["timeframe"])
            if not fresh: pending.pop(sid,None); await q.edit_message_text("🔄 Recheck: setup no longer valid."); return
            fresh["id"]=sid; pending[sid]=fresh
            kb=InlineKeyboardMarkup([[InlineKeyboardButton("✅ APPROVE",callback_data=f"approve:{sid}"),
                                      InlineKeyboardButton("❌ REJECT",callback_data=f"reject:{sid}")],
                                     [InlineKeyboardButton("🔄 RECHECK",callback_data=f"recheck:{sid}")]])
            await q.edit_message_text(render(fresh),reply_markup=kb)
        except Exception as e: await q.edit_message_text(f"Recheck failed: {e}")
        return
    if action=="approve":
        try:
            await c.bot.send_message(chat_id=CHANNEL,text=render(s,"🟢 APPROVED / PUBLISHED"))
            pending.pop(sid,None); await q.edit_message_text(render(s,"🟢 APPROVED / PUBLISHED")+f"\n\n📢 {CHANNEL}")
        except Exception as e:
            await q.edit_message_text(render(s)+f"\n\n⚠️ Publication failed: {e}")

def main():
    if not TOKEN or not ADMIN_CHAT_ID: raise SystemExit("Set TELEGRAM_BOT_TOKEN and ADMIN_CHAT_ID in .env")
    app=Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",start)); app.add_handler(CommandHandler("scan",scan))
    app.add_handler(CommandHandler("status",status)); app.add_handler(CommandHandler("pause",pause))
    app.add_handler(CommandHandler("resume",resume)); app.add_handler(CallbackQueryHandler(callback))
    app.job_queue.run_repeating(scheduled,interval=INTERVAL,first=15)
    app.run_polling()

if __name__=="__main__": main()
