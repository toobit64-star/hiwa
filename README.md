# Hiwacrypto Signal Bot v1

Telegram bot for:
- Spot + Futures signal candidates
- Manual approval before channel publication
- Bitget public market data
- Entry / SL / TP template
- Watchlist scanning
- No trading/order execution in v1

## 1) Install

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## 2) Configure

Copy `.env.example` to `.env` and fill in:

- TELEGRAM_BOT_TOKEN: token from @BotFather
- ADMIN_CHAT_ID: your private Telegram chat ID
- CHANNEL_USERNAME: @hiwacrypto

Do NOT publish your bot token or commit `.env`.

## 3) Add the bot to the channel

Make @fhiwabot an administrator of @hiwacrypto with permission to post messages.

## 4) Run

```bash
python bot.py
```

The bot sends candidate signals to ADMIN_CHAT_ID first.
Nothing is posted to the channel until you press APPROVE.

## Important

This is an analysis/notification bot, not an automated trading bot.
Signal logic in v1 is intentionally conservative and should be tested before relying on it with real money.
