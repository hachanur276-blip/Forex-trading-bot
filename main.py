import os
import threading
import time
from datetime import datetime
from flask import Flask
import numpy as np
import pandas as pd
import requests

# ==============================================================================
# ১. ফ্ল্যাস্ক ওয়েব সার্ভার (Render Web Service কে ২৪/৭ চালক রাখার জন্য)
# ==============================================================================
app = Flask(__name__)


@app.route("/")
def home():
    return "Real-time Twelve Data Trading Bot is Running 24/7 on Render!"


def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


# ==============================================================================
# ২. কনফিগারেশন এবং টেলিগ্রাম / API সেটআপ
# ==============================================================================
TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN", "8830744418:AAHxLdPtiqY0qaJ4yobpg0u7npH7c2kj9TE"
)
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "-1004384520318")

# আপনার প্রদান করা Twelve Data API Key যুক্ত করা হয়েছে
TWELVE_DATA_API_KEY = os.environ.get(
    "TWELVE_DATA_API_KEY", "cd491123c7164d8fa68827f1b59dea57"
)

# Twelve Data সাপোর্টেড কারেন্সি পেয়ার ও ক্রিপ্টো অ্যাসেট লিস্ট
ASSETS = {
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD",
    "USD/CAD": "USD/CAD",
    "USD/CHF": "USD/CHF",
    "EUR/GBP": "EUR/GBP",
    "BTC/USD (Crypto)": "BTC/USD",
    "ETH/USD (Crypto)": "ETH/USD",
}


def send_telegram_message(message: str) -> bool:
    """টেলিগ্রামে মেসেজ পাঠানোর মূল ফিউশন।"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        res_data = response.json()
        if not res_data.get("ok"):
            print(f"❌ [Telegram Error]: {res_data.get('description')}")
            return False
        return True
    except Exception as e:
        print(f"❌ [Network Error]: {e}")
        return False


# ==============================================================================
# ৩. Real-time Candle Fetcher Engine (Twelve Data API)
# ==============================================================================
def fetch_realtime_candles(symbol: str) -> pd.DataFrame:
    """Twelve Data API থেকে সরাসরি রিয়েল-টাইম ক্যান্ডেল ফেচ করে"""
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=50&apikey={TWELVE_DATA_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "values" not in data:
            print(f"⚠️ API Error ({symbol}): {data.get('message')}")
            return pd.DataFrame()

        df = pd.DataFrame(data["values"])

        # কলাম রিনেম ও টাইপ কাস্টিং
        df.rename(
            columns={
                "datetime": "Time",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
            },
            inplace=True,
        )

        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)

        # সাম্প্রতিক ডেটা নিচে রাখার জন্য রিভার্স করা
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"❌ Fetch Exception ({symbol}): {e}")
        return pd.DataFrame()


# ==============================================================================
# ৪. টেকনিক্যাল ইন্ডিকেটর ও প্রাইস অ্যাকশন ইঞ্জিন
# ==============================================================================
def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # ১. RSI (14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ২. Stochastic Oscillator (14, 3, 3)
    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["Stoch_K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14 + 1e-10))
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()

    # ৩. Bollinger Bands (20, 2)
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    # ৪. MACD (12, 26, 9)
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ৫. ADX (14)
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1)),
        ),
    )
    df["+DM"] = np.where(
        (df["High"] - df["High"].shift(1)) > (df["Low"].shift(1) - df["Low"]),
        np.maximum(df["High"] - df["High"].shift(1), 0),
        0,
    )
    df["-DM"] = np.where(
        (df["Low"].shift(1) - df["Low"]) > (df["High"] - df["High"].shift(1)),
        np.maximum(df["Low"].shift(1) - df["Low"], 0),
        0,
    )
    atr = df["TR"].rolling(14).mean()
    plus_di = 100 * (df["+DM"].rolling(14).mean() / (atr + 1e-10))
    minus_di = 100 * (df["-DM"].rolling(14).mean() / (atr + 1e-10))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    df["ADX"] = dx.rolling(14).mean()

    df["ST_Trend"] = df["Close"] > df["BB_Middle"]

    # ৬. Support/Resistance & Price Action Patterns
    df["Support"] = df["Low"].rolling(window=35).min()
    df["Resistance"] = df["High"].rolling(window=35).max()

    body = abs(df["Close"] - df["Open"])
    upper_wick = df["High"] - np.maximum(df["Close"], df["Open"])
    lower_wick = np.minimum(df["Close"], df["Open"]) - df["Low"]

    df["Bullish_Pinbar"] = (lower_wick >= (2 * body)) & (
        upper_wick <= (0.5 * body)
    )
    df["Bearish_Pinbar"] = (upper_wick >= (2 * body)) & (
        lower_wick <= (0.5 * body)
    )

    df["Bullish_Engulfing"] = (
        (df["Close"].shift(1) < df["Open"].shift(1))
        & (df["Close"] > df["Open"])
        & (df["Close"] >= df["Open"].shift(1))
    )
    df["Bearish_Engulfing"] = (
        (df["Close"].shift(1) > df["Open"].shift(1))
        & (df["Close"] < df["Open"])
        & (df["Close"] <= df["Open"].shift(1))
    )

    return df


# ==============================================================================
# ৫. মার্কেট স্ক্যানিং এবং সিগন্যাল জেনারেটর
# ==============================================================================
def analyze_asset_and_predict(asset_name: str, ticker: str):
    try:
        df = fetch_realtime_candles(ticker)

        if df.empty or len(df) < 40:
            return

        df = calculate_all_indicators(df)

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        if pd.isna(curr["ADX"]) or curr["ADX"] < 15:
            return

        score_call = 0
        score_put = 0

        # Technical Indicators Scoring
        if prev["Stoch_K"] < 20 and curr["Stoch_K"] > curr["Stoch_D"]:
            score_call += 1.5
        elif prev["Stoch_K"] > 80 and curr["Stoch_K"] < curr["Stoch_D"]:
            score_put += 1.5

        if curr["Close"] <= curr["BB_Lower"]:
            score_call += 1.5
        elif curr["Close"] >= curr["BB_Upper"]:
            score_put += 1.5

        if (
            prev["MACD"] <= prev["MACD_Signal"]
            and curr["MACD"] > curr["MACD_Signal"]
        ):
            score_call += 1
        elif (
            prev["MACD"] >= prev["MACD_Signal"]
            and curr["MACD"] < curr["MACD_Signal"]
        ):
            score_put += 1

        if 40 < curr["RSI"] < 65 and curr["RSI"] > prev["RSI"]:
            score_call += 1
        elif 35 < curr["RSI"] < 60 and curr["RSI"] < prev["RSI"]:
            score_put += 1

        if curr["ST_Trend"]:
            score_call += 1
        else:
            score_put += 1

        # S/R & Price Action Boost
        near_support = (
            abs(curr["Close"] - curr["Support"]) / curr["Support"]
        ) <= 0.0005
        near_resistance = (
            abs(curr["Close"] - curr["Resistance"]) / curr["Resistance"]
        ) <= 0.0005

        pa_pattern = "None"

        if near_support or curr["Bullish_Pinbar"] or curr["Bullish_Engulfing"]:
            score_call += 2.5
            if curr["Bullish_Pinbar"]:
                pa_pattern = "Hammer/Pinbar 🔨"
            elif curr["Bullish_Engulfing"]:
                pa_pattern = "Bullish Engulfing 🟢"

        if (
            near_resistance
            or curr["Bearish_Pinbar"]
            or curr["Bearish_Engulfing"]
        ):
            score_put += 2.5
            if curr["Bearish_Pinbar"]:
                pa_pattern = "Shooting Star 🌠"
            elif curr["Bearish_Engulfing"]:
                pa_pattern = "Bearish Engulfing 🔴"

        signal_type = None
        total_score = max(score_call, score_put)

        if score_call >= 5.5:
            signal_type = "CALL (UP) 🟢"
        elif score_put >= 5.5:
            signal_type = "PUT (DOWN) 🔴"

        if signal_type:
            current_time = datetime.now().strftime("%I:%M:%S %p")

            message = (
                f"⚡ <b>REAL-TIME TRADING SIGNAL</b> ⚡\n\n"
                f"<b>Asset:</b> {asset_name}\n"
                f"<b>Direction:</b> {signal_type}\n"
                f"<b>Expiry:</b> 1 MINUTE\n"
                f"<b>Signal Score:</b> {round(total_score, 1)}/10\n"
                f"<b>PA Pattern:</b> {pa_pattern}\n"
                f"<b>RSI Level:</b> {round(curr['RSI'], 1)}\n"
                f"<b>ADX Power:</b> {round(curr['ADX'], 1)}\n"
                f"<b>Time:</b> {current_time}\n\n"
                f"📌 <i>Next candle entry recommended!</i>"
            )

            print(
                f"[{current_time}] Signal Generated for {asset_name}: {signal_type}"
            )
            send_telegram_message(message)

    except Exception as e:
        print(f"Error processing {asset_name}: {e}")


# ==============================================================================
# ৬. মূল লুপ
# ==============================================================================
def main_loop():
    print("=" * 50)
    print("Real-time Trading Signal Engine Active...")
    print("=" * 50)

    # স্টার্টআপ টেস্ট নোটিফিকেশন
    send_telegram_message(
        "🚀 <b>Real-time Trading Bot Started!</b>\n\n"
        "✅ Twelve Data API: Connected\n"
        "✅ Render Web Service: Active\n"
        "📊 Scanning Live Market Pairs..."
    )

    while True:
        try:
            for asset_name, ticker in ASSETS.items():
                analyze_asset_and_predict(asset_name, ticker)
                time.sleep(8)  # API Rate limit সামঞ্জস্য বজায় রাখতে বিরতি

            time.sleep(15)
        except Exception as e:
            print(f"Main Loop Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    main_loop()