# Run this once, message your bot /start, then run this script.
# It prints recent Telegram updates so you can find your private chat ID.
import os, requests
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("TELEGRAM_BOT_TOKEN")
data = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10).json()
print(data)
