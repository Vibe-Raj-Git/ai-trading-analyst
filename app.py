import json
import re
import os
import logging
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
#from openai import OpenAI
from flask import Flask, render_template, request, jsonify, session, send_file
#import httpx
import time
import google.generativeai as genai

# Import your trading modules
from new_tradingCopy import analyze, compute_precomputed

# Load environment variables
load_dotenv()

# Remove proxy environment variables to avoid OpenAI client issues
#for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
#    if proxy_var in os.environ:
#        del os.environ[proxy_var]

# Debug: Check if Dhan credentials are loaded
print("=== Dhan Credentials Check ===")
print(f"DHAN_CLIENT_ID: {'SET' if os.getenv('DHAN_CLIENT_ID') else 'NOT SET'}")
print(f"DHAN_PIN: {'SET' if os.getenv('DHAN_PIN') else 'NOT SET'}")
print(f"DHAN_TOTP_SECRET: {'SET' if os.getenv('DHAN_TOTP_SECRET') else 'NOT SET'}")
print("==============================")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ========== SUPPRESS VERBOSE THIRD-PARTY LOGS ==========
# Suppress OpenAI request/response logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Keep your own debug logs
logger.setLevel(logging.DEBUG)

# Flask app initialization
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.config['JSON_AS_ASCII'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Constants
IST = ZoneInfo("Asia/Kolkata")

COLOR_WAIT = "#FFA500"
COLOR_BUY = "#0B7D3E"
COLOR_BUY_STRONG = "#28A745"
COLOR_SELL = "#C1121F"
COLOR_RANGE_BUY = "#7BD38D"
COLOR_RANGE_SELL = "#F25F5C"
COLOR_WARNING = "#FFC107"

# Index symbols and aliases
INDEX_SYMBOLS = {
    "NIFTY50", "NIFTYBANK", "NIFTYNXT50", "NIFTY MIDCAP 150",
    "NIFTY SMLCAP 250", "NIFTY ALPHA 50", "NIFTY200", "NIFTY500",
    "NIFTYIT", "NIFTYFMCG", "NIFTYPHARMA", "NIFTYMETAL", "NIFTYAUTO",
    "NIFTYFINSERVICE", "NIFTY INFRA", "SENSEX", "BANKEX",
}

SYMBOL_ALIASES = {
    "BANKNIFTY": "NIFTYBANK",
    "FINNIFTY": "NIFTYFINSERVICE",
}

NO_PARENT_INDEX_ETFS = {"SILVERBEES", "GOLDBEES"}

IS_DEV = os.getenv("DEV_MODE", "").lower() in ("true", "1", "yes")


# ======================================================
# Helper Functions (copied from original)
# ======================================================
def fmt2(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return x

def fmt_text_prices(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(
        r"\b\d+\.\d{3,}\b",
        lambda m: fmt2(m.group(0)),
        text
    )

def simplify_internal_labels(text: str) -> str:
    if not isinstance(text, str):
        return text
    replacements = {
        "RetailChop": "choppy sideways market",
        "SmartRange": "well-defined range",
        "BearishTrend": "downtrend",
        "BullishTrend": "uptrend",
        "fade": "sell rallies near resistance",
        "Fade": "sell rallies near resistance",
        "PDH": "yesterday's high",
        "PDL": "yesterday's low",
        "regime": "market condition",
        "SmartMoneyTrend": "institutional trend view",
        "Trend_Regime": "trend-timeframe market condition",
        "Setup_Regime": "setup-timeframe market condition",
        "Entry_Regime": "entry-timeframe market condition",
        "SmartMoney": "institutional flow",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def extract_final_strategies_from_output(text: str):
    if not text:
        return None
    idx = text.rfind("STRATEGIES_JSON")
    if idx == -1:
        return None
    tail = text[idx:]
    first_brace = tail.find("{")
    last_brace = tail.rfind("}")
    if first_brace == -1 or last_brace == -1:
        return None
    json_str = tail[first_brace:last_brace + 1]
    try:
        data = json.loads(json_str)
    except Exception:
        return None
    if isinstance(data, dict):
        if "A" in data and "B" in data:
            return data
        if "STRATEGIES_JSON" in data and isinstance(data["STRATEGIES_JSON"], dict):
            inner = data["STRATEGIES_JSON"]
            if "A" in inner and "B" in inner:
                return inner
    return None

def extract_strategies_from_text(text: str):
    """
    Extract strategies from text format (not JSON)
    Parses the strategy section that appears in the raw analysis output
    """
    strategies = {"A": {}, "B": {}}
    
    if not text:
        return None
    
    # Find Strategy A section - handles both formats with and without "Enforced"
    a_match = re.search(
        r"Strategy A:.*?Entry\s*:\s*([\d.]+).*?Stop Loss\s*:\s*([\d.]+).*?Target 1\s*:\s*([\d.]+).*?Target 2\s*:\s*([\d.]+).*?Conviction\s*:\s*(\w+)",
        text, re.DOTALL | re.IGNORECASE
    )
    if a_match:
        strategies["A"] = {
            "entry": a_match.group(1),
            "stop": a_match.group(2),
            "target1": a_match.group(3),
            "target2": a_match.group(4),
            "conviction": a_match.group(5),
            "type": "Pullback"
        }
    
    # Find Strategy B section
    b_match = re.search(
        r"Strategy B:.*?Entry\s*:\s*([\d.]+).*?Stop Loss\s*:\s*([\d.]+).*?Target 1\s*:\s*([\d.]+).*?Target 2\s*:\s*([\d.]+).*?Conviction\s*:\s*(\w+)",
        text, re.DOTALL | re.IGNORECASE
    )
    if b_match:
        strategies["B"] = {
            "entry": b_match.group(1),
            "stop": b_match.group(2),
            "target1": b_match.group(3),
            "target2": b_match.group(4),
            "conviction": b_match.group(5),
            "type": "Range-Edge Fade"
        }
    
    return strategies if strategies["A"] or strategies["B"] else None

def extract_verified_prices(section1_output: str) -> str:
    """
    Parse Section 1 (Full Engine Output) and extract ALL actual prices, 
    indicators, regimes, market structure levels, and F&O metrics.

    Build a comprehensive verified anchor for TRAINER_PROMPT to reference.
    Prevents TRAINER_PROMPT from fabricating ANY prices, indicators, or regimes.
    """
    if not section1_output:
        return ""

    # Common flags
    FLAGS = re.IGNORECASE | re.DOTALL

    # Comprehensive extraction patterns for ALL indicators and levels
    extraction_patterns = {
        # ===== OHLC & Price Levels =====
        "Daily High": r"(?:Daily|DAILY)[^\n]*?High[:\s]+(\d+\.?\d*)",
        "Daily Low": r"(?:Daily|DAILY)[^\n]*?Low[:\s]+(\d+\.?\d*)",
        "Daily Open": r"(?:Daily|DAILY)[^\n]*?Open[:\s]+(\d+\.?\d*)",
        "Daily Close": r"(?:Daily|DAILY)[^\n]*?Close[:\s]+(\d+\.?\d*)",
        "Weekly Close": r"(?:Weekly|WEEKLY)[^\n]*?Close[:\s]+(\d+\.?\d*)",
        "Monthly Close": r"(?:Monthly|MONTHLY)[^\n]*?Close[:\s]+(\d+\.?\d*)",

        # ===== EMAs (All Periods) =====
        "EMA10": r"EMA10[:\s]+(\d+\.?\d*)",
        "EMA20": r"EMA20[:\s]+(\d+\.?\d*)",
        "EMA50": r"EMA50[:\s]+(\d+\.?\d*)",
        "EMA100": r"EMA100[:\s]+(\d+\.?\d*)",
        "EMA200": r"EMA200[:\s]+(\d+\.?\d*)",

        # ===== Momentum Indicators =====
        "RSI14": r"RSI14[:\s]+(\d+\.?\d*)",
        "MACD": r"MACD[:\s]+(\d+\.?\d*)",
        "MACD Signal": r"MACD[_\s]*Signal[:\s]+(\d+\.?\d*)",
        "MACD Histogram": r"MACD[_\s]*(?:Hist|Histogram)[:\s]+(\d+\.?\d*)",
        "Stoch RSI K": r"StochRSI[_\s]*K[:\s]+(\d+\.?\d*)",
        "Stoch RSI D": r"StochRSI[_\s]*D[:\s]+(\d+\.?\d*)",

        # ===== Trend + Volatility =====
        "ADX14": r"ADX14[:\s]+(\d+\.?\d*)",
        "+DI": r"\+DI[:\s]+(\d+\.?\d*)",
        "-DI": r"-DI[:\s]+(\d+\.?\d*)",
        "ATR": r"ATR[:\s]+(\d+\.?\d*)",

        # ===== Volume Indicators =====
        "OBV": r"OBV[:\s]+(\d+\.?\d*)",
        "OBV Slope": r"OBV[_\s]*Slope[:\s]+(\d+\.?\d*)",
        "MFI": r"MFI[:\s]+(\d+\.?\d*)",
        "RVOL": r"RVOL[:\s]+(\d+\.?\d*)",

        # ===== Bands & Channels =====
        "BB Mid": r"BB[_\s]*Mid[:\s]+(\d+\.?\d*)",
        "BB High": r"BB[_\s]*(?:High|Hi)[:\s]+(\d+\.?\d*)",
        "BB Low": r"BB[_\s]*(?:Low|Lo)[:\s]+(\d+\.?\d*)",
        "KC Mid": r"KC[_\s]*Mid[:\s]+(\d+\.?\d*)",
        "KC Upper": r"KC[_\s]*Upper[:\s]+(\d+\.?\d*)",
        "KC Lower": r"KC[_\s]*Lower[:\s]+(\d+\.?\d*)",
        "VWAP": r"VWAP[:\s]+(\d+\.?\d*)",

        # ===== Regimes (Critical!) =====
        "Weekly Regime": r"(?:Weekly|W)[_\s]*Regime[:\s]+([A-Za-z0-9_\-]+)",
        "Daily Regime": r"(?:Daily|D)[_\s]*Regime[:\s]+([A-Za-z0-9_\-]+)",
        "4H Regime": r"(?:4H|H4)[_\s]*Regime[:\s]+([A-Za-z0-9_\-]+)",
        "1H Regime": r"(?:1H|H)[_\s]*Regime[:\s]+([A-Za-z0-9_\-]+)",
        "30M Regime": r"(?:30M|M30)[_\s]*Regime[:\s]+([A-Za-z0-9_\-]+)",
        "15M Regime": r"(?:15M|M15)[_\s]*Regime[:\s]+([A-Za-z0-9_\-]+)",
        "5M Regime": r"(?:5M|M5)[_\s]*Regime[:\s]+([A-Za-z0-9_\-]+)",

        # ===== Range Scores =====
        "Weekly RangeScore": r"(?:Weekly|W)[_\s]*RangeScore[:\s]+(\d+\.?\d*)",
        "Daily RangeScore": r"(?:Daily|D)[_\s]*RangeScore[:\s]+(\d+\.?\d*)",
        "4H RangeScore": r"(?:4H|H4)[_\s]*RangeScore[:\s]+(\d+\.?\d*)",
        "1H RangeScore": r"(?:1H|H)[_\s]*RangeScore[:\s]+(\d+\.?\d*)",
        "30M RangeScore": r"(?:30M|M30)[_\s]*RangeScore[:\s]+(\d+\.?\d*)",
        "15M RangeScore": r"(?:15M|M15)[_\s]*RangeScore[:\s]+(\d+\.?\d*)",
        "5M RangeScore": r"(?:5M|M5)[_\s]*RangeScore[:\s]+(\d+\.?\d*)",

        # ===== Relative Strength (Mansfield & Buckets) =====
        "Daily RS Bucket": r"Daily RS[^\n]*?(?:bucket|Bucket)[:\s]+([A-Za-z0-9_\-]+)",
        "Daily RS Mansfield": r"Daily RS[^\n]*?(?:Mansfield|MANSFIELD)[:\s]+(-?\d+\.?\d*)",
        "Weekly RS Bucket": r"Weekly RS[^\n]*?(?:bucket|Bucket)[:\s]+([A-Za-z0-9_\-]+)",
        "Weekly RS Mansfield": r"Weekly RS[^\n]*?(?:Mansfield|MANSFIELD)[:\s]+(-?\d+\.?\d*)",

        # ===== RSI Divergence (all TFs / personas) =====
        "Daily RSI Divergence Type": r"Daily RSI Divergence Type[:\s]+([\w_]+)",
        "Daily RSI Divergence Strength": r"Daily RSI Divergence Strength[:\s]+(\d+\.?\d*)",
        "Weekly RSI Divergence Type": r"Weekly RSI Divergence Type[:\s]+([\w_]+)",
        "Weekly RSI Divergence Strength": r"Weekly RSI Divergence Strength[:\s]+(\d+\.?\d*)",
        "Monthly RSI Divergence Type": r"Monthly RSI Divergence Type[:\s]+([\w_]+)",
        "Monthly RSI Divergence Strength": r"Monthly RSI Divergence Strength[:\s]+(\d+\.?\d*)",
        "Quarterly RSI Divergence Type": r"Quarterly RSI Divergence Type[:\s]+([\w_]+)",
        "Quarterly RSI Divergence Strength": r"Quarterly RSI Divergence Strength[:\s]+(\d+\.?\d*)",
        "4H RSI Divergence Type": r"4H RSI Divergence Type[:\s]+([\w_]+)",
        "4H RSI Divergence Strength": r"4H RSI Divergence Strength[:\s]+(\d+\.?\d*)",
        "1H RSI Divergence Type": r"1H RSI Divergence Type[:\s]+([\w_]+)",
        "1H RSI Divergence Strength": r"1H RSI Divergence Strength[:\s]+(\d+\.?\d*)",
        "30M RSI Divergence Type": r"30M RSI Divergence Type[:\s]+([\w_]+)",
        "30M RSI Divergence Strength": r"30M RSI Divergence Strength[:\s]+(\d+\.?\d*)",
        "15M RSI Divergence Type": r"15M RSI Divergence Type[:\s]+([\w_]+)",
        "15M RSI Divergence Strength": r"15M RSI Divergence Strength[:\s]+(\d+\.?\d*)",
        "5M RSI Divergence Type": r"5M RSI Divergence Type[:\s]+([\w_]+)",
        "5M RSI Divergence Strength": r"5M RSI Divergence Strength[:\s]+(\d+\.?\d*)",

        # ===== Strategy Entry/Exit Levels =====
        "Strategy A Entry": r"Strategy\s+A[:\s]*Entry[:\s]+(\d+\.?\d*)",
        "Strategy A Stop": r"Strategy\s+A[:\s]*(?:Stop|SL)[:\s]+(\d+\.?\d*)",
        "Strategy A Target": r"Strategy\s+A[:\s]*(?:Target|TP)[:\s]+(\d+\.?\d*)",
        "Strategy A Type": r"Strategy\s+A[:\s]*(?:Type|Style)[:\s]+([^\n]+)",
        "Strategy B Entry": r"Strategy\s+B[:\s]*Entry[:\s]+(\d+\.?\d*)",
        "Strategy B Stop": r"Strategy\s+B[:\s]*(?:Stop|SL)[:\s]+(\d+\.?\d*)",
        "Strategy B Target": r"Strategy\s+B[:\s]*(?:Target|TP)[:\s]+(\d+\.?\d*)",
        "Strategy B Type": r"Strategy\s+B[:\s]*(?:Type|Style)[:\s]+([^\n]+)",

        # ===== Support/Resistance Levels =====
        "S1": r"S1[:\s]+(\d+\.?\d*)",
        "S2": r"S2[:\s]+(\d+\.?\d*)",
        "R1": r"R1[:\s]+(\d+\.?\d*)",
        "R2": r"R2[:\s]+(\d+\.?\d*)",

        # ===== Market Stage (Trend TF) =====
        "Market Stage (Trend TF)": r"MARKET_STAGE_TREND_TF[:\s]+([A-Za-z0-9_\-]+)",

        # ===== DARVAS BOX CONTEXT =====
        "Darvas Box Upper": r"(?:Darvas[_\s]*Box[_\s]*(?:Upper|Resistance)|upper)[:\s]+(\d+\.?\d*)",
        "Darvas Box Lower": r"(?:Darvas[_\s]*Box[_\s]*(?:Lower|Support)|lower)[:\s]+(\d+\.?\d*)",
        "Darvas Box Mid": r"(?:Darvas[_\s]*Box[_\s]*(?:Mid|Midpoint|Pivot)|mid)[:\s]+(\d+\.?\d*)",
        "Darvas Box State": r"(?:[Cc]urrent[_\s]*(?:price|Price).*?[Pp]osition|state)[:\s]*([\w_]+)",
        "Darvas Strength": r"Darvas[_\s]*[Ss]trength[_\s]*[Ss]core[:\s]+(\d+\.?\d*)",
        "Darvas Reliability": r"[Bb]reakout[_\s]*[Rr]eliability[:\s]+([A-Za-z0-9_\-]+)",
        "Darvas Consolidation Bars": r"[Cc]onsolidation[_\s]*[Bb]ars[:\s]+(\d+)",

        # ===== Fibonacci Levels (from swing leg) =====
        "Fib 23.6": r"Fib(?:onacci)?[:\s]+23\.6%[:\s]+(\d+\.?\d*)",
        "Fib 38.2": r"Fib(?:onacci)?[:\s]+38\.2%[:\s]+(\d+\.?\d*)",
        "Fib 50.0": r"Fib(?:onacci)?[:\s]+50\.0%[:\s]+(\d+\.?\d*)",
        "Fib 61.8": r"Fib(?:onacci)?[:\s]+61\.8%[:\s]+(\d+\.?\d*)",
        "Fib 78.6": r"Fib(?:onacci)?[:\s]+78\.6%[:\s]+(\d+\.?\d*)",
        "Fib High": r"Fib(?:onacci)?[_\s]*High[:\s]+(\d+\.?\d*)",
        "Fib Low": r"Fib(?:onacci)?[_\s]*Low[:\s]+(\d+\.?\d*)",

        # ===== F&O / OPTIONS METRICS (from engine narrative) =====
        # Adjust these regexes to match your actual Section 1 labels
        "ATM IV": r"ATM IV[:\s]+(\d+\.?\d*)",
        "ATM IV Change": r"ATM IV change(?:\s*\(today\))?[:\s]+(-?\d+\.?\d*)",
        "PCR": r"PCR[:\s]+(\d+\.?\d*)",
        "PCR Change": r"PCR change(?:\s*\(today\))?[:\s]+(-?\d+\.?\d*)",
        "CE OI Change": r"CE OI change(?:\s*\(today\))?[:\s]+(-?\d+\.?\d*)",
        "PE OI Change": r"PE OI change(?:\s*\(today\))?[:\s]+(-?\d+\.?\d*)",
        "ATM CE Delta": r"ATM CE Delta[:\s]+(-?\d+\.?\d*)",
        "ATM PE Delta": r"ATM PE Delta[:\s]+(-?\d+\.?\d*)",
        "Gamma": r"Gamma[:\s]+(-?\d+\.?\d*)",
        "Vega": r"Vega[:\s]+(-?\d+\.?\d*)",
        "Theta": r"Theta(?:\s*\(time decay\))?[:\s]+(-?\d+\.?\d*)",
        "Futures Direction": r"1H Futures direction[:\s]+([A-Za-z0-9_\-]+)",
        "Futures OI State": r"1H Futures OI state[:\s]+([A-Za-z0-9_\-]+)",
        "Futures Basis": r"Futures basis[:\s]+(-?\d+\.?\d*)",
        "Futures Basis Change": r"Futures basis change(?:\s*\(today\))?[:\s]+(-?\d+\.?\d*)",
    }

    # Extract all values
    extracted_data: dict[str, str] = {}
    for label, pattern in extraction_patterns.items():
        match = re.search(pattern, section1_output, FLAGS)
        if match:
            extracted_data[label] = match.group(1).strip()

    # Build the comprehensive verified anchor
    if not extracted_data:
        return ""

    # Organize by category for readability
    anchor_lines = [
        "=" * 80,
        "VERIFIED DATA FROM ANALYSIS - DO NOT FABRICATE OR CHANGE ANYTHING",
        "=" * 80,
        ""
    ]

    # OHLC
    ohlc_keys = [
        "Daily High", "Daily Low", "Daily Open",
        "Daily Close", "Weekly Close", "Monthly Close"
    ]
    ohlc_data = {k: v for k, v in extracted_data.items() if k in ohlc_keys}
    if ohlc_data:
        anchor_lines.append("OHLC & PRICE LEVELS:")
        for label, value in ohlc_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")

    # EMAs
    ema_data = {k: v for k, v in extracted_data.items() if "EMA" in k}
    if ema_data:
        anchor_lines.append("MOVING AVERAGES (EMAs):")
        for label, value in ema_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")

    # Momentum
    momentum_data = {
        k: v for k, v in extracted_data.items()
        if any(x in k for x in ["RSI", "MACD", "Stoch"])
    }
    if momentum_data:
        anchor_lines.append("MOMENTUM INDICATORS:")
        for label, value in momentum_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")

    # Trend & Volatility
    trend_data = {
        k: v for k, v in extracted_data.items()
        if any(x in k for x in ["ADX", "DI", "ATR"])
    }
    if trend_data:
        anchor_lines.append("TREND & VOLATILITY:")
        for label, value in trend_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")

    # Volume
    volume_data = {
        k: v for k, v in extracted_data.items()
        if any(x in k for x in ["OBV", "MFI", "RVOL"])
    }
    if volume_data:
        anchor_lines.append("VOLUME INDICATORS:")
        for label, value in volume_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")

    # Bands & Channels
    band_data = {
        k: v for k, v in extracted_data.items()
        if any(x in k for x in ["BB ", "BB_", "KC", "VWAP"])
    }
    if band_data:
        anchor_lines.append("BOLLINGER BANDS & KELTNER CHANNELS:")
        for label, value in band_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")

    # Fibonacci
    fib_data = {k: v for k, v in extracted_data.items() if "Fib " in k}
    if fib_data:
        anchor_lines.append("FIBONACCI LEVELS (DO NOT INVENT OR ADJUST):")
        for label, value in fib_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")

    # Regimes
    regime_data = {k: v for k, v in extracted_data.items() if "Regime" in k}
    if regime_data:
        anchor_lines.append("MARKET REGIMES (DO NOT INVENT OR CHANGE):")
        for label, value in regime_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")

    # Range Scores
    range_data = {k: v for k, v in extracted_data.items() if "RangeScore" in k}
    if range_data:
        anchor_lines.append(
            "RANGE SCORES (Use only for descriptive language like 'weak', 'strong', 'balanced'):"
        )
        for label, value in range_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")

    # Relative Strength (Mansfield & Buckets)
    rs_data = {
        k: v for k, v in extracted_data.items()
        if "RS Bucket" in k or "RS Mansfield" in k
    }
    if rs_data:
        anchor_lines.append("RELATIVE STRENGTH (MANSFIELD) – READ-ONLY CONTEXT:")
        for label, value in rs_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")

    # Market Stage (explicit Trend TF)
    stage_val = extracted_data.get("Market Stage (Trend TF)")
    if stage_val:
        anchor_lines.append("MARKET STAGE (TREND TF) – DO NOT INVENT OR CHANGE:")
        anchor_lines.append(f"  • Market Stage (Trend TF): {stage_val}")
        anchor_lines.append("")

    # === RSI DIVERGENCE CONTEXT (ALL TIMEFRAMES) ===
    rsi_div_data = {
        k: v for k, v in extracted_data.items()
        if "RSI Divergence" in k
    }
    if rsi_div_data:
        anchor_lines.append("=" * 80)
        anchor_lines.append("RSI DIVERGENCE CONTEXT (READ-ONLY – DO NOT INVENT):")
        anchor_lines.append("=" * 80)
        tf_order = [
            "Quarterly", "Monthly", "Weekly",
            "Daily", "4H", "1H", "30M", "15M", "5M"
        ]
        for tf in tf_order:
            tf_items = {
                k: v for k, v in rsi_div_data.items()
                if k.startswith(f"{tf} RSI Divergence")
            }
            if not tf_items:
                continue
            anchor_lines.append(f"{tf.upper()} RSI DIVERGENCE:")
            for label, value in tf_items.items():
                anchor_lines.append(f"  • {label}: {value}")
            anchor_lines.append("")
        anchor_lines.append("RSI DIVERGENCE USAGE RULES:")
        anchor_lines.append("  • You may describe divergence ONLY using these exact types and strengths.")
        anchor_lines.append("  • Divergence is SECONDARY to deterministic market conditions (Regime + RangeScore).")
        anchor_lines.append("  • NEVER change Q/MN/W/D or D/M30/M5 bias or condition based on divergence alone.")
        anchor_lines.append("  • Use divergence only as supporting context near PRECOMPUTED structure ")
        anchor_lines.append("    (demand/supply, swings, fib levels, HVN/LVN, yesterday's high/low, etc.).")
        anchor_lines.append("  • NEVER invent new divergence labels or strength scores.")
        anchor_lines.append("")

    # Strategies
    strategy_data = {k: v for k, v in extracted_data.items() if "Strategy" in k}
    if strategy_data:
        anchor_lines.append("TRADING STRATEGIES (EXACT LEVELS - DO NOT CHANGE):")
        for label, value in strategy_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")

    # Support/Resistance
    sr_data = {
        k: v for k, v in extracted_data.items()
        if k in ["S1", "S2", "R1", "R2"]
    }
    if sr_data:
        anchor_lines.append("SUPPORT & RESISTANCE LEVELS:")
        for label, value in sr_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")

    # Darvas Box Context
    darvas_data = {k: v for k, v in extracted_data.items() if "Darvas" in k}
    if darvas_data:
        anchor_lines.append("=" * 80)
        anchor_lines.append("DARVAS BOX CONTEXT (INSTITUTIONAL STRUCTURE - DO NOT INVENT):")
        anchor_lines.append("=" * 80)
        for label, value in darvas_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")
        anchor_lines.append("DARVAS BOX STRATEGY RULES:")
        anchor_lines.append("  • If State is 'below_lower': Price BROKE BELOW support → caution on aggressive longs, favor mean-reversion to lower level")
        anchor_lines.append("  • If State is 'above_upper': Price BROKE ABOVE resistance → favor trend-following continuation longs")
        anchor_lines.append("  • If State is 'inside': Price INSIDE box → use upper as resistance, lower as support, favor range or mean-reversion ideas near the edges")
        anchor_lines.append("  • CRITICAL: Use ONLY the exact upper/lower/mid prices from this section. DO NOT compute new box levels.")
        anchor_lines.append("")

    # ===== OPTIONS & FUTURES METRICS (F&O) =====
    fo_keys = [
        "ATM IV", "ATM IV Change", "PCR", "PCR Change",
        "CE OI Change", "PE OI Change",
        "ATM CE Delta", "ATM PE Delta",
        "Gamma", "Vega", "Theta",
        "Futures Direction", "Futures OI State",
        "Futures Basis", "Futures Basis Change",
    ]
    fo_data = {k: v for k, v in extracted_data.items() if k in fo_keys}
    if fo_data:
        anchor_lines.append("=" * 80)
        anchor_lines.append("OPTIONS & FUTURES METRICS (READ-ONLY – DO NOT INVENT):")
        anchor_lines.append("=" * 80)
        for label, value in fo_data.items():
            anchor_lines.append(f"  • {label}: {value}")
        anchor_lines.append("")
        anchor_lines.append("F&O USAGE RULES:")
        anchor_lines.append("  • You may ONLY refer to these IV, PCR, OI, Greek, and futures values.")
        anchor_lines.append("  • Do NOT invent new IV ranges, PCR thresholds, or strike-wise OI ladders.")
        anchor_lines.append("  • Use these numbers only as context to support or weaken the existing directional bias and risk framing.")
        anchor_lines.append("")

    # CRITICAL RULES
    anchor_lines.extend([
        "=" * 80,
        "CRITICAL RULES FOR TRAINER_PROMPT:",
        "=" * 80,
        "1. EVERY price level mentioned MUST come from the list above.",
        "2. EVERY indicator value mentioned MUST come from the list above.",
        "3. EVERY market condition label (regime) mentioned MUST come from the list above.",
        "4. Do NOT invent any new prices that are not listed above.",
        "5. DO NOT invent new market condition names. You may ONLY use labels that appear in the list above (for example, Bullish, Bearish, Range, SmartRange, RetailChop, etc.).",
        "6. When describing RangeScore, use plain language ONLY: 'weak', 'strong', 'balanced', 'moderate'.",
        "7. DO NOT mention the field names (W_RangeScore, D_RangeScore, etc.) - only use the interpretation.",
        "8. Reference EXACT prices from this list when discussing support, resistance, or levels.",
        "9. If a trader would find a price useful, check this list first before using it.",
        "10. Do NOT write numeric EMA chains like '394.2>390.3>393.4>341.8'. Describe EMAs qualitatively instead (for example, 'short-term EMAs are stacked above longer-term EMAs and price is far above the long-term EMA').",
        "11. Only use phrases like 'LONG-only' or 'SHORT-only' when the analysis explicitly says that such conditions are active (for example, when a Short-only or Long-only rule is clearly stated in the analysis text).",
        "12. When talking about Fibonacci retracements or extensions, use ONLY the Fib levels listed above. Do NOT compute or invent new Fib prices.",
        "13. You may describe Fib zones qualitatively (e.g., 'near 61.8% pullback') but the numbers must match the verified list.",
        "",
        "14. DARVAS BOX RULES:",
        "    a. When discussing the Darvas Box, you MUST use the exact upper/lower/mid levels from the DARVAS BOX CONTEXT section above.",
        "    b. You MUST state clearly: Is price trading INSIDE, ABOVE upper, or BELOW lower the box?",
        "    c. You MUST explain the strategic implication of the current Darvas Box state (from the strategy rules above).",
        "    d. DO NOT compute or invent new Darvas box levels. Use ONLY the verified levels provided.",
        "    e. When price is 'below_lower', highlight this as a breakdown warning and favor mean-reversion ideas back to the lower level.",
        "    f. When price is 'above_upper', highlight this as a breakout confirmation and favor trend-following in bullish context.",
        "    g. Darvas Strength (0-10 score) indicates consolidation quality: <4 weak, 4-7 moderate, >7 strong.",
        "",
        "15. RSI DIVERGENCE RULES:",
        "    a. You may ONLY refer to RSI divergence types/strengths listed in the RSI DIVERGENCE CONTEXT section.",
        "    b. Divergence is supporting context only. It MUST NOT override or contradict any deterministic market condition.",
        "    c. DO NOT invent new divergence labels (e.g., 'super strong hidden bullish'). Use ONLY the precomputed labels.",
        "    d. Use divergence to explain context near PRECOMPUTED structure (e.g., bullish divergence into demand), never to justify trades against the allowed bias.",
        "",
        "16. RELATIVE STRENGTH (RS) RULES:",
        "    a. You may ONLY refer to RS buckets and Mansfield values listed in the RELATIVE STRENGTH section above.",
        "    b. RS is supporting context only. It MUST NOT override or contradict deterministic conditions or allowed trade direction.",
        "    c. DO NOT invent new RS buckets or numeric RS values. Use ONLY the precomputed bucket labels (e.g., StrongOutperform, Outperform, Neutral, Underperform, StrongUnderperform) and Mansfield scores.",
        "    d. Use RS to describe relative strength vs the primary index (e.g., 'Weekly RS shows StrongOutperform'), never to create new price levels or independent trade triggers.",
        "",
        "17. F&O / OPTIONS RULES:",
        "    a. You may ONLY refer to IV, PCR, OI, Greek, and futures values that appear in the OPTIONS & FUTURES METRICS section above.",
        "    b. DO NOT invent any new IV ranges, PCR thresholds, or strike-wise OI distributions that are not explicitly listed.",
        "    c. DO NOT create new Greek values or turn Greeks into exact P&L numbers; use them only as qualitative context for risk and sensitivity.",
        "    d. Use F&O metrics only to support or weaken the existing directional bias and risk framing, never to flip direction or invent new levels.",
        "    e. When you describe IV as low/normal/high or PCR as extreme/neutral, it MUST be consistent with the actual numbers in the OPTIONS & FUTURES METRICS section.",
        "    f. When you interpret PCR_OI you MUST map it correctly to call-heavy vs put-heavy:",
        "       - PCR_OI > 1.0 ⇒ put-heavy positioning (more puts than calls outstanding).",
        "       - PCR_OI < 1.0 ⇒ call-heavy positioning (more calls than puts outstanding).",
        "       - For example, PCR(OI) around 0.74 indicates call-heavy positioning (more calls than puts), which often aligns with covered call writing or traders fading further downside.",
        "       You MUST NOT say \"more puts than calls\" when PCR_OI is below 1.0.",
        "=" * 80,
    ])

    return "\n".join(anchor_lines)

def extract_darvas_direct(text: str) -> dict:
    """Extract Darvas box data directly from debug output format"""
    darvas_data = {}
    
    if not text:
        return darvas_data
    
    # Look for Darvas box in debug format: "DEBUG DARVAS BOX - TF: DAILY"
    # Then capture the following lines with upper, lower, mid, state
    pattern = r"DEBUG DARVAS BOX - TF: DAILY\s+upper:\s+([\d.]+)\s+lower:\s+([\d.]+)\s+mid:\s+([\d.]+)\s+state:\s+(\w+)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        darvas_data["Darvas Box Upper"] = match.group(1)
        darvas_data["Darvas Box Lower"] = match.group(2)
        darvas_data["Darvas Box Mid"] = match.group(3)
        darvas_data["Darvas Box State"] = match.group(4)
        
        # Also capture strength
        strength_match = re.search(r"Darvas Strength:\s+([\d.]+)/10", text, re.IGNORECASE)
        if strength_match:
            darvas_data["Darvas Strength"] = strength_match.group(1)
    
    return darvas_data

def extract_rsi_divergence_from_raw(text: str) -> str:
    """Extract RSI divergence from raw debug output"""
    rsi_anchor = ""
    
    # Look for patterns like: "DEBUG RSI_DIVERGENCE WEEKLY: type=bearish, strength=1.2396929307391815"
    pattern = r"DEBUG RSI_DIVERGENCE\s+(\w+):\s+type=(\w+),\s+strength=([\d.]+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    for match in matches:
        tf = match[0]
        div_type = match[1]
        strength = float(match[2])
        
        if div_type != "none" and strength > 0:
            if rsi_anchor == "":
                rsi_anchor = "=" * 80 + "\n"
                rsi_anchor += "RSI DIVERGENCE CONTEXT (READ-ONLY – DO NOT INVENT):\n"
                rsi_anchor += "=" * 80 + "\n"
            
            # Map timeframe to readable format
            tf_map = {
                "5M": "5-Minute", "15M": "15-Minute", "30M": "30-Minute",
                "1H": "1-Hour", "4H": "4-Hour", "DAILY": "Daily",
                "WEEKLY": "Weekly", "MONTHLY": "Monthly", "QUARTERLY": "Quarterly"
            }
            tf_readable = tf_map.get(tf.upper(), tf)
            
            rsi_anchor += f"  • {tf_readable} RSI Divergence: {div_type.upper()} (strength: {strength:.2f})\n"
    
    if rsi_anchor:
        rsi_anchor += "\n"
    
    return rsi_anchor

def extract_fibonacci_from_raw(text: str) -> str:
    """Extract Fibonacci levels from raw output (from build_fib_levels_from_leg)"""
    fib_anchor = ""
    
    # Look for Fibonacci levels in the debug output
    # They may appear as "DEBUG: Fibonacci levels: {'23.6': 259.80, ...}"
    # Or in the GANN metrics section
    
    fib_patterns = {
        "23.6": r"['\"]23\.6['\"]:\s*([\d.]+)",
        "38.2": r"['\"]38\.2['\"]:\s*([\d.]+)",
        "50.0": r"['\"]50\.0['\"]:\s*([\d.]+)",
        "61.8": r"['\"]61\.8['\"]:\s*([\d.]+)",
        "78.6": r"['\"]78\.6['\"]:\s*([\d.]+)",
    }
    
    fib_values = {}
    for level, pattern in fib_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fib_values[level] = match.group(1)
    
    if fib_values:
        fib_anchor = "=" * 80 + "\n"
        fib_anchor += "FIBONACCI LEVELS (DO NOT INVENT OR ADJUST):\n"
        fib_anchor += "=" * 80 + "\n"
        for level, value in fib_values.items():
            fib_anchor += f"  • {level}%: {value}\n"
        fib_anchor += "\n"
    
    return fib_anchor

def extract_ema_stack_from_raw(text: str) -> str:
    """Extract EMA stack description from debug output"""
    ema_anchor = ""
    
    # Look for EMA stack comment
    pattern = r"DEBUG EMA STACK DAILY\s+->\s+(.+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        ema_anchor = "=" * 80 + "\n"
        ema_anchor += "EMA STACK ANALYSIS:\n"
        ema_anchor += "=" * 80 + "\n"
        ema_anchor += f"  • Daily: {match.group(1)}\n"
        
        # Also get weekly
        weekly_match = re.search(r"DEBUG EMA STACK WEEKLY\s+->\s+(.+)", text, re.IGNORECASE)
        if weekly_match:
            ema_anchor += f"  • Weekly: {weekly_match.group(1)}\n"
        ema_anchor += "\n"
    
    return ema_anchor

def extract_gann_metrics_for_anchor(gann_metrics: dict) -> str:
    """Extract GANN metrics for the anchor"""
    if not gann_metrics:
        return ""
    
    gann_anchor = "=" * 80 + "\n"
    gann_anchor += "GANN METRICS (SUPPORTING CONTEXT ONLY – Do NOT override primary regimes):\n"
    gann_anchor += "=" * 80 + "\n"
    
    # Weekly patterns
    weekly = gann_metrics.get("weekly_patterns", {})
    if weekly.get("friday_weekly_high"):
        gann_anchor += f"  • Friday made weekly high → Next week bias: {weekly.get('next_week_bias', 'N/A')}\n"
    if weekly.get("friday_weekly_low"):
        gann_anchor += f"  • Friday made weekly low → Next week bias: {weekly.get('next_week_bias', 'N/A')}\n"
    
    # Day of week patterns
    dow = gann_metrics.get("day_of_week_patterns", {})
    if dow.get("wednesday_high_in_downtrend"):
        gann_anchor += f"  • Wednesday high → Downtrend signal ({dow.get('confidence', 0)}% confidence)\n"
    
    # Monthly patterns
    monthly = gann_metrics.get("monthly_patterns", {})
    if monthly.get("double_top"):
        gann_anchor += f"  • Monthly Double Top detected ({monthly.get('gap_months', 0)} months gap) → {monthly.get('signal', 'N/A')}\n"
    if monthly.get("triple_top"):
        gann_anchor += f"  • Monthly Triple Top detected ({monthly.get('gap_months', 0)} months gap) → {monthly.get('signal', 'N/A')}\n"
    
    # Quarterly breakout
    qtr = gann_metrics.get("quarterly_breakout", {})
    if qtr.get("breakdown_below"):
        gann_anchor += f"  • Quarterly close below previous quarter's low → Bearish trend reversal\n"
    
    gann_anchor += "\n"
    return gann_anchor

return generate_fallback_trainer_explanation(
    precomputed=precomputed,
    regimes=regimes,
    strategies={},
    current_price=current_price,
    supports=[],
    resistances=[],
    market_stage=market_stage,
    clean_output=clean_output,
    raw_output=raw_output  # ✅ CRITICAL - must be passed
)

def compute_quick_action(mode: str, regimes: dict | None, darvas_state: str | None = None):
    """
    Derive Quick Action (bias + message + color) from persona-level regimes.
    Uses simplified, beginner-friendly language.

    NOTE: darvas_state is expected to be pre-filtered for proximity
    (None when USE_DARVAS_FOR_TRAINER is False).
    """

    # Helper: enforce persona constraints (long-only for investing/positional)
    def clamp_bias_for_mode(bias_value: str, mode_value: str) -> str:
        mode_l = (mode_value or "").strip().lower()
        if mode_l in {"investing", "positional"}:
            if bias_value == "SELL":
                return "RANGE-SELL"
        return bias_value

    # Defaults
    bias = "WAIT"
    color = COLOR_WAIT
    msg = (
        "No clear trade setup right now. "
        "Wait for price to reach key levels or for a clearer opportunity."
    )

    if not isinstance(regimes, dict):
        bias = clamp_bias_for_mode(bias, mode)
        return {"bias": bias, "color": color, "message": msg}

    trend_regime = (regimes.get("Trend_Regime") or "").strip().lower()
    setup_regime = (regimes.get("Setup_Regime") or "").strip().lower()
    entry_regime = (regimes.get("Entry_Regime") or "").strip().lower()

    if not trend_regime:
        bias = clamp_bias_for_mode(bias, mode)
        return {"bias": bias, "color": color, "message": msg}

    mode_lower = (mode or "").strip().lower()
    darvas_state = (darvas_state or "").strip().lower() if darvas_state else None

    # --- 1) Trend = RetailChop: NEW softer rules ---
    if trend_regime == "retailchop":
        # If both lower timeframes clearly align, still give a directional bias
        if setup_regime == "bullish" and entry_regime == "bullish":
            bias = "RANGE-BUY"
            color = COLOR_RANGE_BUY
            msg = (
                "Bigger picture is choppy, but both mid and lower timeframes "
                "are bullish. Prefer buying dips near strong support, "
                "use smaller size and tighter risk."
            )
        elif setup_regime == "bearish" and entry_regime == "bearish":
            bias = "RANGE-SELL"
            color = COLOR_RANGE_SELL
            msg = (
                "Bigger picture is choppy, but both mid and lower timeframes "
                "are bearish. Prefer selling bounces near strong resistance, "
                "use smaller size and tighter risk."
            )
        else:
            # Truly messy: keep hard no-trade
            bias = "WAIT"
            color = COLOR_WAIT
            msg = (
                "Market is choppy and unpredictable right now. "
                "Best to stay out and wait for clearer direction."
            )

        bias = clamp_bias_for_mode(bias, mode)
        return {"bias": bias, "color": color, "message": msg}

    # --- 2) Pure trend environments (clean Bullish / Bearish) ---

    if trend_regime == "bullish":
        # Check for price breaking below support (warning signal)
        if darvas_state == "below_lower":
            bias = "RANGE-BUY"
            color = COLOR_WARNING  # Amber - WARNING
            msg = (
                "⚠️ Uptrend is active but price just broke below recent support. "
                "This could be a buying opportunity near support OR a warning sign. "
                "Only buy at the support level with tight stop-loss."
            )
            bias = clamp_bias_for_mode(bias, mode)
            return {"bias": bias, "color": color, "message": msg}

        # Strong bullish setup
        if setup_regime == "bullish":
            # Upgrade if price broke above resistance
            if darvas_state == "above_upper":
                bias = "BUY"
                color = COLOR_BUY_STRONG  # Bright green
                msg = (
                    "💪 Strong buy signal! Price broke above recent resistance in an uptrend. "
                    "Look for buying opportunities on small dips. Use stops below the breakout level."
                )
            else:
                bias = "BUY"
                color = COLOR_BUY  # Dark green
                msg = (
                    "Strong buy bias. Focus on buying opportunities only. "
                    "Buy when price dips to support or breaks above resistance."
                )

            if entry_regime in {"range", "retailchop"}:
                msg = (
                    "Buy bias is active, but short-term action is choppy. "
                    "Wait for a clear bounce from support or a strong breakout before entering."
                )

            if mode_lower in {"investing", "positional"} and bias in {"BUY", "SELL"}:
                bias = "RANGE-BUY" if bias == "BUY" else "RANGE-SELL"

            bias = clamp_bias_for_mode(bias, mode)
            return {"bias": bias, "color": color, "message": msg}

        # Bullish trend but mixed intraday
        if setup_regime in {"range", "smartrange", ""}:
            bias = "RANGE-BUY"
            color = COLOR_RANGE_BUY  # Light green
            msg = (
                "Bigger timeframe is bullish, but short-term is mixed. "
                "Look for buying opportunities near strong support levels. Avoid shorting."
            )
            bias = clamp_bias_for_mode(bias, mode)
            return {"bias": bias, "color": color, "message": msg}

        # Conflicting signals
        msg = (
            "Bigger timeframe is bullish but short-term is bearish. "
            "Wait until both align before trading."
        )
        bias = clamp_bias_for_mode("WAIT", mode)
        return {"bias": bias, "color": COLOR_WAIT, "message": msg}

    if trend_regime == "bearish":
        # Check for price breaking below support
        if darvas_state == "below_lower":
            # Only call it a "strong sell" when all TFs are bearish (clean trend)
            if setup_regime == "bearish" and entry_regime == "bearish":
                bias = "SELL"
                color = COLOR_SELL  # Dark red
                msg = (
                    "💪 Strong sell signal! Price broke below recent support in a clean downtrend. "
                    "Look for selling opportunities on small bounces. Use stops above the breakdown level."
                )
            else:
                # Daily Bearish but intraday in Range/SmartRange → Accumulation / oversold zone
                bias = "RANGE-SELL"
                color = COLOR_RANGE_SELL  # Light red
                msg = (
                    "Bigger timeframe is bearish, but price is now below recent support and "
                    "intraday price action is ranging. Focus only on cautious sell-on-rally "
                    "setups near strong resistance. Avoid chasing breakdowns; standing aside is fine."
                )

            bias = clamp_bias_for_mode(bias, mode)
            return {"bias": bias, "color": color, "message": msg}

        # Strong bearish setup
        if setup_regime == "bearish":
            bias = "SELL"
            color = COLOR_SELL  # Dark red
            msg = (
                "Strong sell bias. Focus on selling opportunities only. "
                "Sell when price bounces to resistance or breaks below support."
            )

            if entry_regime in {"range", "retailchop"}:
                msg = (
                    "Sell bias is active, but short-term action is choppy. "
                    "Wait for a clear rejection from resistance or a strong breakdown before entering."
                )

            if mode_lower in {"investing", "positional"} and bias in {"BUY", "SELL"}:
                bias = "RANGE-BUY" if bias == "BUY" else "RANGE-SELL"

            bias = clamp_bias_for_mode(bias, mode)
            return {"bias": bias, "color": color, "message": msg}

        # Bearish trend but mixed intraday
        if setup_regime in {"range", "smartrange", ""}:
            bias = "RANGE-SELL"
            color = COLOR_RANGE_SELL  # Light red
            msg = (
                "Bigger timeframe is bearish, but short-term is mixed. "
                "Look for selling opportunities near strong resistance levels. Avoid buying."
            )
            bias = clamp_bias_for_mode(bias, mode)
            return {"bias": bias, "color": color, "message": msg}

        # Conflicting signals
        msg = (
            "Bigger timeframe is bearish but short-term is bullish. "
            "Wait until both align before trading."
        )
        bias = clamp_bias_for_mode("WAIT", mode)
        return {"bias": bias, "color": COLOR_WAIT, "message": msg}

    # --- 3) Sideways/Range markets ---
    if trend_regime in {"range", "smartrange"}:
        # Price inside consolidation box
        if darvas_state == "inside":
            if setup_regime == "bullish" or entry_regime == "bullish":
                bias = "RANGE-BUY"
                color = COLOR_RANGE_BUY
                msg = (
                    "Market moving sideways with slight bullish tilt. "
                    "Buy ONLY at the bottom of the range; avoid buying in the middle. "
                    "Target: top of the range."
                )
            elif setup_regime == "bearish" or entry_regime == "bearish":
                bias = "RANGE-SELL"
                color = COLOR_RANGE_SELL
                msg = (
                    "Market moving sideways with slight bearish tilt. "
                    "Sell ONLY at the top of the range; avoid selling in the middle. "
                    "Target: bottom of the range."
                )
            else:
                bias = "WAIT"
                color = COLOR_WAIT
                msg = (
                    "Market stuck in sideways range with no clear direction. "
                    "Only trade at the very top (sell) or very bottom (buy) of the range."
                )

            bias = clamp_bias_for_mode(bias, mode)
            return {"bias": bias, "color": color, "message": msg}

        # Price broke above range
        if darvas_state == "above_upper":
            if setup_regime in {"bullish", "smartrange"}:
                bias = "BUY"
                color = COLOR_BUY_STRONG
                msg = (
                    "Breakout confirmed! Price broke above the recent range. "
                    "Look for buying opportunities. Avoid new shorts unless price falls back into the range."
                )
            else:
                bias = "RANGE-BUY"
                color = COLOR_RANGE_BUY
                msg = (
                    "Price broke above recent range, but signals are mixed. "
                    "Only buy with strong confirmation."
                )

            if mode_lower in {"investing", "positional"} and bias in {"BUY", "SELL"}:
                bias = "RANGE-BUY" if bias == "BUY" else "RANGE-SELL"

            bias = clamp_bias_for_mode(bias, mode)
            return {"bias": bias, "color": color, "message": msg}

        # Price broke below range
        if darvas_state == "below_lower":
            if setup_regime in {"bearish", "smartrange"}:
                bias = "SELL"
                color = COLOR_SELL
                msg = (
                    "Breakdown confirmed! Price broke below the recent range. "
                    "Look for selling opportunities. Avoid new buys unless price bounces back into the range."
                )
            else:
                bias = "RANGE-SELL"
                color = COLOR_RANGE_SELL
                msg = (
                    "Price broke below recent range, but signals are mixed. "
                    "Only sell with strong confirmation."
                )

            if mode_lower in {"investing", "positional"} and bias in {"BUY", "SELL"}:
                bias = "RANGE-BUY" if bias == "BUY" else "RANGE-SELL"

            bias = clamp_bias_for_mode(bias, mode)
            return {"bias": bias, "color": color, "message": msg}

        # Generic range message (no Darvas info)
        if setup_regime == "bullish" or entry_regime == "bullish":
            bias = "RANGE-BUY"
            color = COLOR_RANGE_BUY
            msg = (
                "Market moving sideways with bullish tilt. "
                "Buy only near strong support at the bottom of the range. "
                "Don't chase prices in the middle."
            )
        elif setup_regime == "bearish" or entry_regime == "bearish":
            bias = "RANGE-SELL"
            color = COLOR_RANGE_SELL
            msg = (
                "Market moving sideways with bearish tilt. "
                "Sell only near strong resistance at the top of the range. "
                "Don't chase prices in the middle."
            )
        else:
            bias = "WAIT"
            color = COLOR_WAIT
            msg = (
                "Market stuck sideways with no clear tilt. "
                "Only trade at clear range edges or stay on the sidelines."
            )

        bias = clamp_bias_for_mode(bias, mode)
        return {"bias": bias, "color": color, "message": msg}

    # Fallback
    bias = clamp_bias_for_mode(bias, mode)
    return {"bias": bias, "color": color, "message": msg}

def get_trainer_key(stock, mode):
    return f"trainer_{stock}_{mode}"

# ======================================================
# Trainer Explanation Generation with Full Prompt
# ======================================================
def generate_trainer_explanation(clean_output, raw_output, persona_key, is_index, precomputed, verified_prices_anchor, fo_snapshot, fo_decision_snapshot, rs_snapshot, gann_metrics, regimes=None, market_stage=None):
    """Generate trainer explanation using Gemini (primary) - converts technical to beginner-friendly"""

    # ✅ CHECK API KEY FIRST - NO HARDCODING
    GENAI_API_KEY = os.environ.get("GENAI_API_KEY") or ""
    
    if not GENAI_API_KEY or GENAI_API_KEY == "":
        print("DEBUG WARNING: GENAI_API_KEY not set! Using fallback explanation.")
        
        # Get current price based on persona
        if persona_key in ("positional", "position"):
            current_price = precomputed.get("MONTHLY", {}).get("indicators", {}).get("MN_Close", 0)
        elif persona_key in ("investing", "investment"):
            current_price = precomputed.get("QUARTERLY", {}).get("indicators", {}).get("Q_Close", 0)
        elif persona_key == "swing":
            current_price = precomputed.get("WEEKLY", {}).get("indicators", {}).get("W_Close", 0)
        else:
            current_price = precomputed.get("DAILY", {}).get("indicators", {}).get("D_Close", 0)
        
        return generate_fallback_trainer_explanation(
            precomputed=precomputed,
            regimes=regimes,
            strategies={},
            current_price=current_price,
            supports=[],
            resistances=[],
            market_stage=market_stage,
            raw_output=raw_output  # ✅ Pass raw output, not cleaned
        )
    # ========== DEBUG: Check gann_metrics received ==========
    print("=" * 80)
    print("DEBUG INSIDE generate_trainer_explanation: gann_metrics received:")
    print("=" * 80)
    if gann_metrics:
        print(f"gann_metrics keys: {list(gann_metrics.keys())}")
        weekly = gann_metrics.get("weekly_patterns", {})
        print(f"weekly_patterns: {weekly}")
        monthly = gann_metrics.get("monthly_patterns", {})
        print(f"monthly_patterns: {monthly}")
    else:
        print("❌ WARNING: gann_metrics is EMPTY or None inside generate_trainer_explanation!")
    print("=" * 80)

    # If market_stage not provided, try to get from precomputed
    if market_stage is None:
        # Try to get from Daily market structure
        daily_ms = precomputed.get("DAILY", {}).get("market_structure", {})
        market_stage = daily_ms.get("market_stage")
        
        # If still None, try Weekly
        if market_stage is None:
            weekly_ms = precomputed.get("WEEKLY", {}).get("market_structure", {})
            market_stage = weekly_ms.get("market_stage")
        
        # If still None, log warning and leave as None (don't invent a value)
        if market_stage is None:
            print("DEBUG WARNING: market_stage not found in precomputed! Available keys:", list(precomputed.keys()))
            # Don't assign a default - let it be None
        else:
            print(f"DEBUG: market_stage retrieved from precomputed: {market_stage}")
    else:
        print(f"DEBUG: market_stage provided as parameter: {market_stage}")

    # Set timeframe explainer based on persona
    if persona_key == "intraday":
        tf_explainer = "Daily → 30m → 5m"
    elif persona_key == "swing":
        tf_explainer = "Weekly → Daily → 4H"
    elif persona_key in ("positional", "position"):
        tf_explainer = "Monthly → Weekly → Daily"
    elif persona_key in ("fno", "fo"):
        tf_explainer = "Daily → 30m → 5m"
    elif persona_key in ("investing", "investment"):
        tf_explainer = "Quarterly → Monthly → Weekly"
    else:
        tf_explainer = "Weekly → Daily → 4H"
    
    # ========== ADD MARKET STAGE CONTEXT FOR THE PROMPT ==========
    market_stage_context = ""
    if market_stage:
        market_stage_context = f"""
═══════════════════════════════════════════════════════════════════════════════
MARKET_STAGE_TREND_TF: {market_stage}
═══════════════════════════════════════════════════════════════════════════════

CRITICAL INSTRUCTION - USE THIS EXACT VALUE:
The above MARKET_STAGE_TREND_TF is the EXACT value calculated by the system's institutional logic.
You MUST use this exact value in your explanation. DO NOT change it or invent a different stage.

- If the value is "Accumulation", write "Accumulation" (base building / consolidation phase)
- If the value is "Advancing", write "Advancing" (markup / uptrend phase)
- If the value is "Distribution", write "Distribution" (top building phase)
- If the value is "Declining", write "Declining" (markdown / downtrend phase)

Example: "The Trend timeframe is tagged in the analysis as being in an {market_stage} phase..."

"""
        print(f"DEBUG: Added market_stage_context to prompt: {market_stage}")
    else:
        market_stage_context = """
═══════════════════════════════════════════════════════════════════════════════
NOTE: No specific MARKET_STAGE_TREND_TF was provided by the analysis.
If the analysis text mentions a stage (Accumulation/Advancing/Distribution/Declining), use that value.
If not, describe the market condition qualitatively without inventing a stage name.
═══════════════════════════════════════════════════════════════════════════════

"""
        print("DEBUG WARNING: No market_stage available to add to prompt")
    
    # ========== BUILD ADDITIONAL ANALYTICS ANCHOR ==========
    additional_anchor_lines = []
    
    # 1. Daily Indicators with Bollinger Status
    daily_ind = precomputed.get("DAILY", {}).get("indicators", {})
    if daily_ind:
        additional_anchor_lines.append("=" * 80)
        additional_anchor_lines.append("ADDITIONAL VERIFIED DATA (FROM PRECOMPUTED):")
        additional_anchor_lines.append("=" * 80)
        additional_anchor_lines.append("")
        
        # Bollinger Bands with status
        bb_hi = daily_ind.get('D_BB_hi')
        bb_lo = daily_ind.get('D_BB_lo')
        close = daily_ind.get('D_Close')
        if bb_hi and bb_lo and close:
            additional_anchor_lines.append("BOLLINGER BANDS (Daily, 20-period, 3σ):")
            additional_anchor_lines.append(f"  • Upper Band: {bb_hi:.2f}")
            additional_anchor_lines.append(f"  • Lower Band: {bb_lo:.2f}")
            if close > bb_hi:
                additional_anchor_lines.append(f"  • Status: Price ABOVE upper band → STRETCHED/OVERBOUGHT")
            elif close < bb_lo:
                additional_anchor_lines.append(f"  • Status: Price BELOW lower band → WASHED OUT/OVERSOLD")
            else:
                pct = (close - bb_lo) / (bb_hi - bb_lo) * 100
                additional_anchor_lines.append(f"  • Status: Inside bands ({pct:.0f}% from lower to upper)")
            additional_anchor_lines.append("")
    
    # 2. EMA Stack Comments
    weekly_ind = precomputed.get("WEEKLY", {}).get("indicators", {})
    daily_ema_comment = daily_ind.get("D_EMA_comment", "")
    weekly_ema_comment = weekly_ind.get("W_EMA_comment", "")
    if daily_ema_comment or weekly_ema_comment:
        additional_anchor_lines.append("EMA STACK ANALYSIS:")
        if daily_ema_comment:
            additional_anchor_lines.append(f"  • Daily: {daily_ema_comment}")
        if weekly_ema_comment:
            additional_anchor_lines.append(f"  • Weekly: {weekly_ema_comment}")
        additional_anchor_lines.append("")
    
    # 3. RSI Divergence (Multi-Timeframe) - FIXED to ensure it appears
    divergences = []
    tf_map = {
        "5M": "5-Minute", "15M": "15-Minute", "30M": "30-Minute",
        "1H": "1-Hour", "4H": "4-Hour", "DAILY": "Daily", "WEEKLY": "Weekly"
    }
    for tf, tf_label in tf_map.items():
        tf_data = precomputed.get(tf, {})
        if tf_data:
            ms_block = tf_data.get("market_structure", {})
            div_type = ms_block.get("rsi_divergence_type")
            div_strength = ms_block.get("rsi_divergence_strength")
            if div_type and div_type != "none" and div_strength > 0:
                divergences.append(f"  • {tf_label}: {div_type.upper()} (strength: {div_strength:.2f})")
    
    if divergences:
        additional_anchor_lines.append("RSI DIVERGENCE (MULTI-TIMEFRAME) - EARLY WARNING SIGNALS:")
        additional_anchor_lines.extend(divergences)
        additional_anchor_lines.append("  • RULE: Divergence is SECONDARY to primary market condition. Use as confirmation only.")
        additional_anchor_lines.append("")
    
    # 4. Darvas Box with Strength and Actual Levels
    for tf in ["DAILY", "WEEKLY", "MONTHLY"]:
        tf_data = precomputed.get(tf, {})
        ms_block = tf_data.get("market_structure", {})
        darvas = ms_block.get("darvas_box", {})
        if darvas and darvas.get("is_valid_classical_darvas"):
            darvas_strength = ms_block.get("darvas_strength", {})
            additional_anchor_lines.append(f"DARVAS BOX ({tf} TF) - INSTITUTIONAL STRUCTURE:")
            additional_anchor_lines.append(f"  • Upper (Resistance): {darvas.get('upper', 'N/A'):.2f}")
            additional_anchor_lines.append(f"  • Lower (Support): {darvas.get('lower', 'N/A'):.2f}")
            additional_anchor_lines.append(f"  • Mid (50% Pivot): {darvas.get('mid', 'N/A'):.2f}")
            additional_anchor_lines.append(f"  • Current Price Position: {darvas.get('state', 'inside')}")
            additional_anchor_lines.append(f"  • Darvas Strength Score: {darvas_strength.get('darvas_strength', 0):.1f}/10")
            additional_anchor_lines.append(f"  • Consolidation Quality: {darvas_strength.get('consolidation_quality', 'Unknown')}")
            additional_anchor_lines.append(f"  • Breakout Reliability: {darvas_strength.get('breakout_reliability', 'Low')}")
            additional_anchor_lines.append(f"  • Consolidation Bars: {darvas.get('consolidation_bars', 0)}")
            additional_anchor_lines.append("")
            break
    
    # 5. Fibonacci Levels
    for tf in ["DAILY", "WEEKLY", "MONTHLY"]:
        tf_data = precomputed.get(tf, {})
        ms_block = tf_data.get("market_structure", {})
        fib = ms_block.get("fib_levels", {})
        if fib:
            additional_anchor_lines.append("FIBONACCI LEVELS (FROM SWING LEG - DO NOT INVENT):")
            for level, price in fib.items():
                additional_anchor_lines.append(f"  • {level}%: {price:.2f}")
            additional_anchor_lines.append("")
            break
    
    # 6. Relative Strength (Mansfield)
    if rs_snapshot:
        daily_bucket = rs_snapshot.get("daily_bucket", "Neutral")
        daily_mansfield = rs_snapshot.get("daily_mansfield")
        weekly_bucket = rs_snapshot.get("weekly_bucket", "Neutral")
        weekly_mansfield = rs_snapshot.get("weekly_mansfield")
        
        additional_anchor_lines.append("RELATIVE STRENGTH (MANSFIELD) - VS PRIMARY INDEX:")
        if daily_mansfield is not None:
            additional_anchor_lines.append(f"  • Daily: {daily_bucket} (Mansfield: {daily_mansfield:+.3f})")
        else:
            additional_anchor_lines.append(f"  • Daily: {daily_bucket}")
        if weekly_mansfield is not None:
            additional_anchor_lines.append(f"  • Weekly: {weekly_bucket} (Mansfield: {weekly_mansfield:+.3f})")
        else:
            additional_anchor_lines.append(f"  • Weekly: {weekly_bucket}")
        additional_anchor_lines.append("  • RS helps prioritise stronger vs weaker names")
        additional_anchor_lines.append("")
    
    # 7. GANN Metrics
    if gann_metrics:
        additional_anchor_lines.append("=" * 80)
        additional_anchor_lines.append("GANN METRICS (SUPPORTING CONTEXT ONLY - DO NOT OVERRIDE PRIMARY REGIME):")
        additional_anchor_lines.append("=" * 80)
        
        # Weekly patterns
        weekly = gann_metrics.get("weekly_patterns", {})
        if weekly.get("friday_weekly_high"):
            additional_anchor_lines.append(f"  • Friday made weekly high → Next week bias: {weekly.get('next_week_bias', 'Bullish')} ({weekly.get('confidence', 65)}% confidence)")
        if weekly.get("friday_weekly_low"):
            additional_anchor_lines.append(f"  • Friday made weekly low → Next week bias: {weekly.get('next_week_bias', 'Bearish')} ({weekly.get('confidence', 65)}% confidence)")
        
        # Day of week patterns
        dow = gann_metrics.get("day_of_week_patterns", {})
        if dow.get("wednesday_high_in_downtrend"):
            additional_anchor_lines.append(f"  • Wednesday made weekly high → Downtrend signal ({dow.get('confidence', 85)}% confidence)")
        if dow.get("tuesday_low_in_uptrend"):
            additional_anchor_lines.append(f"  • Tuesday made weekly low → Uptrend signal ({dow.get('confidence', 85)}% confidence)")
        
        # Monthly patterns
        monthly = gann_metrics.get("monthly_patterns", {})
        if monthly.get("double_top"):
            additional_anchor_lines.append(f"  • Monthly DOUBLE TOP detected ({monthly.get('gap_months', 0)} months gap) → {monthly.get('signal', 'BEARISH')} signal")
        if monthly.get("triple_top"):
            additional_anchor_lines.append(f"  • Monthly TRIPLE TOP detected ({monthly.get('gap_months', 0)} months gap) → {monthly.get('signal', 'BEARISH')} signal")
        if monthly.get("double_bottom"):
            additional_anchor_lines.append(f"  • Monthly DOUBLE BOTTOM detected ({monthly.get('gap_months', 0)} months gap) → {monthly.get('signal', 'BULLISH')} signal")
        if monthly.get("triple_bottom"):
            additional_anchor_lines.append(f"  • Monthly TRIPLE BOTTOM detected ({monthly.get('gap_months', 0)} months gap) → {monthly.get('signal', 'BULLISH')} signal")
        
        # Quarterly breakout
        qtr = gann_metrics.get("quarterly_breakout", {})
        if qtr.get("breakout_above"):
            additional_anchor_lines.append(f"  • Quarterly breakout above {qtr.get('previous_quarter_high')} → BULLISH reversal signal")
        if qtr.get("breakdown_below"):
            additional_anchor_lines.append(f"  • Quarterly breakdown below {qtr.get('previous_quarter_low')} → BEARISH reversal signal")
        
        # Breakout patterns
        breakout = gann_metrics.get("breakout_patterns", {})
        if breakout.get("four_week_high_break"):
            additional_anchor_lines.append(f"  • 4-Week high broken at {breakout.get('four_week_high_break')} → Higher prices expected")
        if breakout.get("four_week_low_break"):
            additional_anchor_lines.append(f"  • 4-Week low broken at {breakout.get('four_week_low_break')} → Lower prices expected")
        if breakout.get("three_day_high_signal"):
            additional_anchor_lines.append(f"  • 3-day high broken → Expect 4th day surge, stop at {breakout.get('stop_gann')}")
        
        # Correction ratios
        corr = gann_metrics.get("correction_ratios", {})
        if corr.get("correction_ratio_detected"):
            additional_anchor_lines.append(f"  • {corr.get('correction_ratio_detected')} ratio detected ({corr.get('consecutive_up_days')} up days) → Expected {corr.get('expected_correction_days')} days correction")
        if corr.get("deeper_correction_warning"):
            additional_anchor_lines.append("  • Deeper correction detected → Trend change possible")
        
        # Volume signals
        vol = gann_metrics.get("volume_signals", {})
        if vol.get("volume_spike_detected"):
            additional_anchor_lines.append(f"  • {vol.get('spike_magnitude')}x volume spike in consolidation → Trend change signal ({vol.get('signal_strength')} strength)")
        
        # 30 DMA break
        ma_break = gann_metrics.get("ma_break", {})
        if ma_break.get("ma_break_signal"):
            additional_anchor_lines.append(f"  • {ma_break.get('consecutive_days_below')} consecutive days below 30 DMA → Correction expected")
        
        # 100% Rise Resistance
        hundred_pct = gann_metrics.get("hundred_percent_resistance", {})
        if hundred_pct.get("one_hundred_percent_level"):
            status = "Near resistance" if hundred_pct.get("is_near_resistance") else "Not near"
            additional_anchor_lines.append(f"  • 100% Rise Level: {hundred_pct.get('one_hundred_percent_level')} from low {hundred_pct.get('key_level')} → {status}")
        
        # 50% Sell Zone
        fifty_pct = gann_metrics.get("fifty_percent_sell_zone", {})
        if fifty_pct.get("fifty_percent_level"):
            status = "Below 50% level → Not suitable for investment" if fifty_pct.get("is_below_50_percent") else "Above 50% level → Suitable for investment"
            additional_anchor_lines.append(f"  • 50% Zone: {fifty_pct.get('fifty_percent_level')} from high {fifty_pct.get('last_high')} → {status}")
        
        additional_anchor_lines.append("")
    
    # 8. Smart Money Concepts (Order Blocks, Liquidity, FVG)
    # Extract from DAILY market structure
    daily_ms = precomputed.get("DAILY", {}).get("market_structure", {})
    
    # Order Blocks
    order_blocks = daily_ms.get("order_blocks", [])
    if order_blocks:
        ob_summary = []
        for ob in order_blocks[-3:]:  # Last 3 order blocks
            if isinstance(ob, dict):
                ob_type = ob.get("type", "unknown")
                ob_price = ob.get("price", "N/A")
                if isinstance(ob_price, tuple):
                    ob_summary.append(f"{ob_type} at {ob_price[0]:.2f}-{ob_price[1]:.2f}")
                elif isinstance(ob_price, (int, float)):
                    ob_summary.append(f"{ob_type} at {ob_price:.2f}")
        if ob_summary:
            additional_anchor_lines.append("ORDER BLOCKS (INSTITUTIONAL LEVELS):")
            additional_anchor_lines.append(f"  • Recent: {', '.join(ob_summary)}")
            additional_anchor_lines.append("")
    
    # Liquidity Pools
    liquidity = daily_ms.get("liquidity", [])
    if liquidity:
        liq_summary = []
        for liq in liquidity[-3:]:
            if isinstance(liq, dict):
                liq_type = liq.get("type", "unknown")
                liq_price = liq.get("price", "N/A")
                liq_summary.append(f"{liq_type} at {liq_price:.2f}")
        if liq_summary:
            additional_anchor_lines.append("LIQUIDITY POOLS (STOP HUNT ZONES):")
            additional_anchor_lines.append(f"  • Recent: {', '.join(liq_summary)}")
            additional_anchor_lines.append("")
    
    # Fair Value Gaps (FVG)
    fvg = daily_ms.get("fvg", [])
    if fvg:
        fvg_summary = []
        for f in fvg[-3:]:
            if isinstance(f, dict):
                fvg_type = f.get("type", "unknown")
                fvg_low = f.get("low", "N/A")
                fvg_high = f.get("high", "N/A")
                fvg_summary.append(f"{fvg_type} gap {fvg_low:.2f}-{fvg_high:.2f}")
        if fvg_summary:
            additional_anchor_lines.append("FAIR VALUE GAPS (PRICE INEFFICIENCIES):")
            additional_anchor_lines.append(f"  • Recent: {', '.join(fvg_summary)}")
            additional_anchor_lines.append("")
    
    additional_anchor = "\n".join(additional_anchor_lines)

    # ========== VERIFY additional_anchor HAS ALL REQUIRED SECTIONS ==========
    print("=" * 80)
    print("DEBUG: additional_anchor verification:")
    print("=" * 80)
    
    required_sections = [
        ("GANN", "GANN METRICS"),
        ("RSI DIVERGENCE", "RSI DIVERGENCE"),
        ("BOLLINGER", "BOLLINGER BANDS"),
        ("FIBONACCI", "FIBONACCI LEVELS"),
        ("RELATIVE STRENGTH", "RELATIVE STRENGTH"),
        ("DARVAS", "DARVAS BOX")
    ]
    
    for section_name, section_marker in required_sections:
        if section_marker in additional_anchor:
            print(f"✅ {section_name}: FOUND in additional_anchor")
        else:
            print(f"❌ {section_name}: NOT FOUND in additional_anchor")
    
    print(f"Total additional_anchor length: {len(additional_anchor)} chars")
    print("=" * 80)
    
    # ========== DEBUG: Check if GANN data is in additional_anchor ==========
    print("=" * 80)
    print("DEBUG: additional_anchor content check:")
    print("=" * 80)
    if "GANN" in additional_anchor:
        print("✅ GANN section found in additional_anchor")
        # Print first 500 chars of GANN section
        gann_start = additional_anchor.find("GANN METRICS")
        if gann_start != -1:
            print(additional_anchor[gann_start:gann_start+500])
    else:
        print("❌ WARNING: GANN section NOT found in additional_anchor!")
    
    if "RSI DIVERGENCE" in additional_anchor:
        print("✅ RSI DIVERGENCE section found in additional_anchor")
    else:
        print("❌ WARNING: RSI DIVERGENCE section NOT found in additional_anchor!")
    
    if "BOLLINGER" in additional_anchor:
        print("✅ BOLLINGER section found in additional_anchor")
    else:
        print("❌ WARNING: BOLLINGER section NOT found in additional_anchor!")
    
    print(f"Total additional_anchor length: {len(additional_anchor)} chars")
    print("=" * 80)

    # Add Keltner Channels analysis
    kc_upper = daily_ind.get('D_KC_upper')
    kc_lower = daily_ind.get('D_KC_lower')
    kc_mid = daily_ind.get('D_KC_mid')
    kc_tight = daily_ind.get('D_KC_tight', False)
    if kc_upper and kc_lower:
        additional_anchor_lines.append("KELTNER CHANNELS (Daily, EMA20 ± 2×ATR):")
        additional_anchor_lines.append(f"  • Upper: {kc_upper:.2f}")
        additional_anchor_lines.append(f"  • Middle: {kc_mid:.2f}")
        additional_anchor_lines.append(f"  • Lower: {kc_lower:.2f}")
        additional_anchor_lines.append(f"  • Compression Flag (KC_tight): {kc_tight}")
        if kc_tight:
            additional_anchor_lines.append("  • → Volatility compression detected - watch for breakout")
        additional_anchor_lines.append("")
    
    # Add Supply/Demand Zones from market structure
    daily_ms = precomputed.get("DAILY", {}).get("market_structure", {})
    supply_zones = daily_ms.get("supply_zones", [])
    demand_zones = daily_ms.get("demand_zones", [])
    
    if supply_zones:
        additional_anchor_lines.append("SUPPLY ZONES (Resistance - Institutional Selling):")
        for sz in supply_zones[-3:]:  # Last 3 zones
            if isinstance(sz, dict):
                additional_anchor_lines.append(f"  • {sz.get('high', 'N/A'):.2f} - {sz.get('low', 'N/A'):.2f}")
        additional_anchor_lines.append("")
    
    if demand_zones:
        additional_anchor_lines.append("DEMAND ZONES (Support - Institutional Buying):")
        for dz in demand_zones[-3:]:
            if isinstance(dz, dict):
                additional_anchor_lines.append(f"  • {dz.get('high', 'N/A'):.2f} - {dz.get('low', 'N/A'):.2f}")
        additional_anchor_lines.append("")
    
    # Add Order Blocks
    order_blocks = daily_ms.get("order_blocks", [])
    if order_blocks:
        additional_anchor_lines.append("ORDER BLOCKS (Institutional Levels):")
        for ob in order_blocks[-3:]:
            if isinstance(ob, dict):
                ob_type = ob.get("type", "unknown")
                ob_price = ob.get("price", "N/A")
                if isinstance(ob_price, tuple):
                    additional_anchor_lines.append(f"  • {ob_type.upper()} OB: {ob_price[0]:.2f} - {ob_price[1]:.2f}")
                elif isinstance(ob_price, (int, float)):
                    additional_anchor_lines.append(f"  • {ob_type.upper()} OB: {ob_price:.2f}")
        additional_anchor_lines.append("")
    
    # Build clauses (same as before - keeping all your existing prompt text)
    ignore_rvol_clause = ""
    if is_index:
        ignore_rvol_clause = (
            "- If the symbol is an index (like NIFTY50, NIFTYBANK, NIFTYNXT50, "
            "NIFTY MIDCAP 150, NIFTY200, NIFTY500, NIFTY SMLCAP 250, NIFTYIT, NIFTYFMCG, NIFTYPHARMA, "
            "NIFTYMETAL, NIFTY SMLCAP 250, NIFTY ALPHA 50, NIFTYAUTO, NIFTYFINSERVICE, NIFTY INFRA, SENSEX, BANKEX), ignore RVOL, "
            "VWAP, and MFI values in your reasoning and do not mention RVOL, VWAP, "
            "or MFI in the explanation.\n"
        )
    
    lower_tf_indicator_clause = ""
    if persona_key == "swing":
        lower_tf_indicator_clause = (
            "- For Swing, do NOT quote any numeric indicator values on the 4H timeframe. "
            "You may describe 4H behaviour qualitatively (e.g., '4H momentum is cooling' or "
            "'4H volatility is elevated') but without numbers. Use 4H only for structure and "
            "timing (order blocks, liquidity, FVGs, swings, supply/demand, HVN/LVN).\n"
        )
    else:
        lower_tf_indicator_clause = ""

    persona_label_map = {
        "intraday": "Intraday", "swing": "Swing", "positional": "Positional",
        "fno": "F&O", "fo": "F&O", "investing": "Investing"
    }
    persona_label = persona_label_map.get(persona_key, persona_key)

    print("=" * 80)
    print("DEBUG: In generate_trainer_explanation - verified_prices_anchor:")
    print("=" * 80)
    if verified_prices_anchor and len(verified_prices_anchor) > 0:
        print(verified_prices_anchor[:2000])
        print(f"Length: {len(verified_prices_anchor)} characters")
    else:
        print("WARNING: verified_prices_anchor is EMPTY!")
    print("=" * 80)

    swing_clause = ""
    if persona_key == "swing":
        swing_clause = """
- For the Swing persona, use phrases such as "short-leaning" or "short-favoured" when Weekly and Daily market conditions plus indicators suggest a bearish bias, and reserve literal "SHORT-only" language ONLY when the analysis explicitly states that short-only conditions are in force. Tactical long mean-reversion from strong support is allowed if it does not violate the higher timeframe rules.

- For Swing mode, when describing the Daily timeframe:
- If Daily RSI is above 70 OR the Daily close is above the Daily upper
Bollinger Band (3 standard deviations), you MUST explicitly call the
Daily overbought / stretched / exhausted and link this to
mean-reversion or profit-protection ideas (for example, "selling bounces
off resistance" or "taking profits on stretched longs"), rather than
new aggressive trend-following longs.
"""

    # F&O clause (keep as is)
    fo_clause = ""
    if persona_key in ("fno", "fo"):
        fo_clause = """
- If F&O metrics are present from the OPTIONS & FUTURES METRICS / FO_METRICS snapshot,
you MUST include a compact F&O context block that explains IV, PCR, OI, Greeks,
volatility skew, net positioning, and futures behaviour WITHOUT inventing any new numbers.

IV:
- Use ONLY the provided ATM IV (for calls/puts or combined) and its change today.
- Classify IV as low / normal / high using a simple banding, for example:
- IV < 15  → low
- 15–25    → normal
- > 25     → high
- If both call and put IV are present, say whether they are aligned (both in same band)
or skewed (different bands), and map this ONLY to trade style and risk:
- Both high    → volatile environment; use tighter risk and be careful chasing options.
- Both low     → breakout reliability can be lower; favour pullbacks/mean-reversion.
- Both normal  → standard breakout or pullback structures are acceptable.
- Skew         → options on the higher-IV side are relatively expensive and in higher demand.

Term Structure:
- Note whether term structure is "front_elevated" (near-term uncertainty), "normal_contango", or "flat".

Volatility Skew:
- If volatility skew data exists, note the skew type (put_skew/call_skew/neutral) and strength (mild/strong/extreme):
- put_skew (puts more expensive) = fear/hedging dominant → reinforces bearish bias
- call_skew (calls more expensive) = greed/optimism → reinforces bullish bias
- neutral = balanced sentiment

PCR & OI:
- Use ONLY the PCR value and PCR change from the snapshot.
- Mention PCR once and interpret using these bands:
- PCR ≥ 1.5            → extreme bullish positioning (put-heavy).
- 1.2 ≤ PCR < 1.5      → strong bullish positioning.
- 0.8 ≤ PCR < 1.2      → balanced / neutral.
- 0.6 ≤ PCR < 0.8      → strong bearish positioning.
- PCR < 0.6            → extreme bearish positioning (call-heavy).
- Note OI trend (build_up/unwinding/mixed) to see if positions are being added or closed.
- Use CE/PE OI change ONLY to say whether call-side or put-side positions are building up or unwinding.
- State clearly that PCR/OI are CONTEXT ONLY; they can upgrade/downgrade conviction but MUST NOT override Daily direction or change any price levels.

Net Positioning (Delta Bias):
- If net delta bias is available (bullish/bearish/neutral), use it as additional confirmation of institutional positioning.
- Bullish delta bias + bullish trend = stronger conviction; bearish delta bias + bullish trend = caution.

Gamma Exposure:
- Note gamma exposure level (low/moderate/high/extreme):
- High/Extreme gamma = delta changes rapidly, high whipsaw risk → recommend tighter stops, smaller size, or spread strategies
- Low gamma = more stable, allow wider stops

Liquidity Grade:
- Note liquidity grade (excellent/good/fair/poor):
- Poor/Fair liquidity = avoid large positions, expect slippage, be selective with strikes
- Excellent/Good liquidity = normal execution

Volume Momentum:
- Note volume momentum (rising/flat/falling):
- Rising volume = confirms breakouts
- Falling volume = warns of fakeouts

Greeks (Delta, Gamma, Vega, Theta):
- Use ONLY the ATM Delta values, Gamma, Vega, and Theta from the snapshot.
- Explain them qualitatively, for example:
- Delta → how directional / sensitive the option is to the underlying.
- Gamma → how quickly Delta can change on intraday moves (link to gamma exposure above).
- Vega  → how sensitive option prices are to IV changes.
- Theta → how much time decay hurts long option buyers if price stays flat.
- DO NOT compute or quote any new Greek numbers and DO NOT turn Greeks into exact profit/loss amounts.

1H Futures (direction, OI state, basis, trend strength, participation quality):
- If 1H futures data is present, explain whether long_buildup / short_buildup / short_covering / long_unwinding SUPPORT or WARN against the Daily + 30m conditions.
- Note trend strength (strong/moderate/weak) and participation quality (high/normal/low).
- Use futures data ONLY to upgrade or downgrade conviction (for example, from LOW to MEDIUM or MEDIUM to HIGH), never to flip direction or invent levels.

Risk Profile (from FO_DECISION):
- If risk profile is provided (aggressive/moderate/conservative), calibrate your recommendations accordingly:
- aggressive → can consider naked options, wider stops
- moderate → prefer spreads, defined risk
- conservative → smaller size, tighter stops, avoid low-liquidity strikes
"""
    else:
        fo_clause = ""

    fo_regime_clause = ""
    if persona_key in ("fno", "fo"):
        fo_regime_clause = """
- When summarising bias for F&O, you MUST anchor your wording to the deterministic market conditions from the analysis:
- Use the Daily condition (D_Regime) as the absolute direction lock (for example, SHORT-only if Bearish with a clean trend).
- Use the 30m condition (M30_Regime) only to strengthen or weaken that bias (trend-supporting vs mixed range).
- Use the 5m condition (M5_Regime) only as an execution filter (aligned vs conflicting micro-structure).
- You MUST NOT contradict these conditions in your wording. If the analysis says D_Regime is Bearish,
you cannot call the big-picture bias "bullish" or "strong buy".
- If FO_DECISION exists, you MUST mention fo_bias and fo_conviction once in the explanation, and ensure your narrative never contradicts them.
"""
    else:
        fo_regime_clause = ""

    hide_rangescore_clause = ""
    if persona_key in ("fno", "fo"):
        hide_rangescore_clause = """
- In this rewritten explanation, you may describe how clean or noisy a trend/range is in plain language
(for example, "well-defined trend", "choppy range"), but you MUST NOT mention internal field names like
D_RangeScore, M30_RangeScore, or M5_RangeScore explicitly. These fields may appear in the raw engine analysis,
but your rewritten explanation should only translate them into simple English.
"""
    else:
        hide_rangescore_clause = """
- You may use the internal RangeScore fields to judge whether a market condition is weak, strong, or neutral,
but you MUST NOT mention any field name like W_RangeScore, D_RangeScore, H4_RangeScore, or their numeric values
in the explanation.
- Do NOT write sentences like "the RangeScore is 4.9" or "1H RangeScore is high". Instead, translate these
into plain qualitative language.
- Always describe range quality only in simple terms such as "weak range", "well-defined range",
"choppy sideways market", or "mixed environment", without exposing any internal scores or numbers.
"""

    big_picture_fo_extra = ""
    if persona_key in ("fno", "fo"):
        big_picture_fo_extra = """
- If FO_METRICS / OPTIONS & FUTURES METRICS exists, in your big-picture section you MUST:
- Mention PCR/OI positioning once (bullish / bearish / neutral) and say whether it SUPPORTS or CONFLICTS with the Daily bias.
- Mention whether IV is low / normal / high, and tie that directly to aggression and risk (for example, "volatility is high, so use tighter risk and avoid overleveraging on options").
- Note volatility skew type and what it says about market sentiment.
- Mention net delta bias and gamma exposure level.
- Briefly note if futures OI behaviour (long build-up vs short build-up vs short covering vs long unwinding) is in line with, or against, the directional view from Daily + 30m.
"""
    else:
        big_picture_fo_extra = ""

    darvas_clause = ""
    if persona_key in ("swing", "intraday", "fno", "fo", "positional", "investing"):
        darvas_clause = """
- If the analysis mentions a Darvas-style box or a recent trading range built from the last 3–4 swing highs and lows, you MUST treat it as a key structural feature:
- Call it a "Darvas box", "recent price box", or "current trading range" in simple language.
- Explicitly state the approximate upper and lower levels of this box using the exact prices from the analysis (for example, "a box between 25623 and 25730").
- Say clearly whether price is currently:
- trading INSIDE this box,
- trading ABOVE the top of this box, or
- trading BELOW the bottom of this box,
matching the analysis text.
- When price is inside the box, emphasise range-edge mean-reversion ideas at the box edges and warn against chasing the middle of the range.
- When price is above the upper edge of the box in a bullish context, describe this as a breakout / continuation above the recent box and explain how this favours breakout or trend-following longs, with risk controls.
- When price is below the lower edge of the box in a bearish context, describe this as a breakdown below the recent box and explain how this favours breakdown continuation or warns against new aggressive longs.
- You MUST NOT invent any new box levels or midpoints; only use the exact support/resistance levels and mid-range levels that already appear in the analysis text.
""" 

    fib_clause = """
- If the analysis lists Fibonacci retracement levels (such as 23.6%, 38.2%, 50%, 61.8%, 78.6% or a Fib high/low), you MUST:
- Treat them as higher-timeframe reference zones, not as new levels you can adjust.
- Use ONLY the exact Fib prices from the analysis when you mention discount/premium zones.
- Explain in simple terms how price is positioned relative to these bands, for example:
- "price is pulling back into the 38.2–61.8% retracement zone inside the bigger uptrend"
- "price is stretching beyond the 78.6% area, so new entries are late and riskier"
- Connect these Fib zones to structure that is already in the analysis (support/resistance, Darvas box edges, supply/demand, or swing highs/lows), without inventing any new confluences.
""" 

    # RS clause
    rs_clause = ""
    d_bucket = rs_snapshot.get("daily_bucket")
    w_bucket = rs_snapshot.get("weekly_bucket")
    
    def human_rs(bucket):
        if not bucket:
            return ""
        b = str(bucket)
        if "StrongOutperform" in b:
            return "strongly outperforming the index"
        if "Outperform" in b:
            return "outperforming the index"
        if "StrongUnderperform" in b:
            return "strongly underperforming the index"
        if "Underperform" in b:
            return "underperforming the index"
        return "moving broadly in line with the index"
    
    d_text = human_rs(d_bucket)
    w_text = human_rs(w_bucket)
    
    if is_index:
        rs_clause = ""
    else:
        if d_text or w_text:
            rs_parts = []
            if w_text:
                rs_parts.append(f"- Weekly RS vs index: {w_text}.")
            if d_text:
                rs_parts.append(f"- Daily RS vs index: {d_text}.")
            rs_clause = """
- Use Relative Strength (RS) information as read-only context:
  {lines}
  - RS helps you prioritise stronger vs weaker names but MUST NOT override the higher timeframe market condition or justify trades against it.
  - Do NOT invent any new RS labels or values; only describe RS in plain English using the given buckets.
""".format(lines="\n  ".join(rs_parts))
        else:
            rs_clause = """
- If Relative Strength (RS) is present in the verified data, briefly explain whether the stock is outperforming, lagging, or moving broadly in line with its primary index.
  - Use RS only as confirmation or a ranking tool (stronger vs weaker), not as a source of new levels or independent triggers.
"""

    final_checklist_block = """
**FINAL CHECKLIST:**
Before you finish, ensure your rewrite:
✓ Sounds like advice from a trusted trading mentor, not a report
✓ Uses "you" and "your" to speak directly to the trader at least 5 times
✓ Includes at least 2-3 "why this matters" explanations
✓ Has varied sentence lengths (mix short punchy ones with longer explanatory ones)
✓ Contains 1-2 relatable analogies or real-world comparisons
✓ Addresses trader emotions or common mistakes at least once
✓ Ends each major section with a clear takeaway or bottom-line statement
✓ Feels engaging enough that someone would want to read the whole thing
✓ Uses ZERO technical jargon (no regime, fade, PDH, BOS, CHOCH, FVG, OB)
✓ Every section starts with an engaging hook or transition

**CRITICAL REMINDER:**
- NO internal field names in explanation (RetailChop → "choppy sideways market")
- NO technical abbreviations (PDH → "yesterday's high")
- NO word "fade" (use "sell near resistance" or "buy near support")
- NO word "regime" (use "market condition" or "trend state")
- SPEAK like a mentor, not like a textbook
- DO NOT repeat the same phrase or "Why This Matters" more than 3 times total
- Keep the explanation concise and focused on actionable trading advice
- If you find yourself repeating, stop and move to the next section

**MANDATORY ANALYTICS TO INCLUDE (IF PRESENT IN THE DATA ABOVE):**
1. **GANN SIGNALS** - If GANN_METRICS section shows any signals, you MUST mention:
   - Friday weekly high/low signals with confidence percentage
   - Monthly double/triple top/bottom patterns
   - 3-day high break signals (if present)
   - Example: "GANN shows Friday made weekly high (65% confidence), suggesting bullish bias next week"
   
2. **RSI DIVERGENCE** - If RSI DIVERGENCE section shows any divergences, you MUST mention:
   - Which timeframe (30M, 4H, Daily, Weekly) and type (bullish/bearish)
   - Example: "30-minute chart shows bearish divergence, warning of a potential reversal"

3. **BOLLINGER BANDS** - If Bollinger status is present, you MUST mention:
   - Whether price is above upper band (overbought), below lower band (oversold), or inside bands
   - Example: "Price is trading near the upper Bollinger Band, indicating stretched conditions"

4. **DARVAS BOX** - If Darvas box is present and within proximity, you MUST mention:
   - Upper and lower levels, current position (inside/above/below), and strength
   - Example: "Price is inside a Darvas box between 22182 and 26373 with 4.5/10 strength"

**YOU MUST INCLUDE THESE IN YOUR NARRATIVE. DO NOT SKIP THEM.**

"""

    base_trainer_header = f"""
{verified_prices_anchor}
{additional_anchor}
{darvas_clause}
{fib_clause}
{rs_clause}
"""

    trainer_prompt = (
        market_stage_context +      # ✅ MAKE SURE THIS LINE IS PRESENT
        base_trainer_header
        + "\n"
        + lower_tf_indicator_clause
        + "\n"
        + swing_clause
        + "\n"
        + fo_regime_clause
        + "\n"
        + ignore_rvol_clause
        + "\n"
        + hide_rangescore_clause
        + "\n"
        + big_picture_fo_extra
        + "\n"
        + final_checklist_block
    )

    if persona_key in ("fno", "fo") and (fo_snapshot or fo_decision_snapshot):
        if fo_snapshot:
            trainer_prompt += "\n" + fo_snapshot + "\n"
        if fo_decision_snapshot:
            trainer_prompt += fo_decision_snapshot + "\n"
        trainer_prompt += fo_clause

    # Continue with the rest of the prompt (the long one you already have)
    trainer_prompt += f"""

**YOUR ROLE:**
You are a professional trading strategist delivering clear, actionable market analysis. Your goal is to provide precise guidance that traders can execute without confusion.

**COMMUNICATION PRINCIPLES:**

1. **Direct and Authoritative:**
- State conclusions clearly: "The market is in a downtrend. Here's your game plan."
- Use imperative language: "Watch this level", "Avoid these trades", "Set stops here"
- Be concise: Every sentence should deliver value. No fluff, no fillers.

2. **Action-Oriented:**
- Tell traders WHAT to do and WHEN to do it
- Use clear directives: "Buy above X", "Sell below Y", "Wait for confirmation at Z"
- Focus on execution: "Your entry is at...", "Your stop goes at...", "Targets are..."

3. **Specific Levels Always:**
- Every trade idea MUST include exact price levels
- Example: "Buy pullbacks near 23120.35 support, with stop at 23067.60, target at 23272.40"
- No vague statements like "near support" without the actual number

4. **IF-THEN Structure (MANDATORY):**
- Frame every trade setup as conditional: "If price does X, then do Y"
- Example: "If price holds above 23120, expect a push to 23272. If it breaks below, stand aside"
- This removes ambiguity and gives clear decision points

5. **Risk Management First:**
- Always state stop-loss levels with rationale
- Mention position sizing where appropriate
- Highlight invalidation levels: "Below this level, the setup is invalid"

6. **Concise Sections:**
- Use bullet points for action items
- Keep paragraphs to 2-3 sentences max
- Eliminate repetitive explanations

7. **Clear Takeaways:**
- End each section with a crisp summary: "Bottom line: ...", "Key point: ...", "Remember: ..."
- Use emphasis strategically: "The MOST important level to watch is...", "Whatever you do, DON'T..."

**LANGUAGE RULES (STRICT):**
- NEVER use: "RetailChop", "SmartRange", "BearishTrend", "BullishTrend"
→ Use: "choppy", "range-bound", "downtrend", "uptrend"
- NEVER use: "fade", "fading"
→ Use: "sell into strength" or "buy into weakness"
- NEVER use: "PDH", "PDL", "BOS", "CHOCH", "FVG", "OB", "regime"
→ Use: "yesterday's high/low", "trend change", "reversal", "price gap", "supply/demand zone", "market condition"

**SECTION FORMAT (STRICT):**

1) **Bias & Key Levels**
- State the directional bias in 1 sentence
- List 2-3 key levels with their significance
- Example: "Bearish bias. Resistance at 23272.4. Support at 23120.35."

2) **Trade Setup**
- For the allowed direction only
- Format: "IF price does X, THEN entry at Y, stop at Z, target at A/B"
- Include position sizing guidance

3) **What to Avoid**
- Explicitly state what not to do
- Example: "DO NOT buy at current levels. DO NOT chase breakdowns."

4) **Action Plan (Bullet Points)**
- Each timeframe: specific price levels to watch
- Example:
    • **Trend (Daily)**: Watch 23120 support. If broken, bearish continuation.
    • **Setup (30M)**: Look for rejection at 23272.4.
    • **Entry (5M)**: Wait for bounce off 23120 with buyers.

5) **Risk & Invalidation**
- Stop-loss levels
- Conditions that void the trade

6) **One-Liner Summary**
- Single sentence recap of the game plan

**TONE:**
- Professional, not academic
- Confident, not speculative
- Direct, not conversational
- Every sentence must be actionable

**Remember: Traders need to know exactly what to do, when to do it, and what invalidates the trade. No stories. No analogies. Just action.**

Rewrite the following stock analysis in a clear, professional {persona_label} trading tone in less than 1000 words.

Goals (VERY IMPORTANT):
- Treat this explanation as a forward-looking game plan, NOT a backward-looking report.
- For every key level or structure you mention, answer: "What is the most likely next move if price reacts here?"
- Use clear IF–THEN language, for example:
- "If price holds above <support>, the next likely move is a push toward <resistance>."
- "If price fails to break above <resistance>, expect a pullback toward <support> before any new up-move."
- Keep the multi-timeframe logic: {tf_explainer} (trend → setup → execution). Explain {tf_explainer} trends in simple, direct language.
- Stay fully aligned with the given market conditions, structures, and levels in the analysis. Do NOT invent any new prices, indicators, or structures.
- If the analysis mentions a Darvas-style box or a recent trading range built from the last swing highs and lows, you may refer to it simply as a "recent price box" or "current trading range".
- You MUST NOT invent any new box levels or midpoints; only use the exact support/resistance levels and mid-range levels that already appear in the analysis text.
- When price is already trading ABOVE a strong Darvas-style price box on the higher timeframe(s) in {tf_explainer}, you MUST avoid telling traders to chase fresh highs. Prefer language such as "wait for pullbacks toward clearly defined support levels from this analysis (for example, the nearest Daily or Trend-timeframe support around <exact level>) before considering new longs."
- When you talk about EMAs, avoid dumping long numeric chains like "394.2 > 390.3 > 393.4 > 341.8". Prefer clear descriptions such as "short-term EMAs are stacked above longer-term EMAs" or "price is trading far above the long-term EMA, so the move is extended", and only mention specific EMA levels when they genuinely help and are present in the analysis.
{swing_clause}
- When you refer to any numeric indicator (EMA, RSI, ADX, MACD, etc.), your wording MUST be consistent with its actual value in the analysis. Do NOT say things like "below 20", "above 70", "very weak", or "very strong" if that does not match the number.

{fo_regime_clause}
{ignore_rvol_clause}
{hide_rangescore_clause}

- When RSI divergence is listed in the verified data (for example, "Daily RSI Divergence Type" or "Weekly RSI Divergence Type"):
- Treat it as an EARLY WARNING or CONFIRMATION signal, not a standalone trigger.
- Always tie divergence to PRECOMPUTED structure (demand/supply zones, swings, Fib bands, Darvas box edges, HVN/LVN), for example:
- "bullish RSI divergence into a support zone" or "bearish RSI divergence near resistance".
- For bullish divergence at or just below support:
- Say that sellers are losing momentum and downside pressure is weakening. The next likely path is either:
    - a bounce toward the nearest resistance or EMA band, OR
    - a sideways pause if the higher-timeframe condition is still strongly bearish.
- For bearish divergence at or just above resistance:
- Say that buyers are losing momentum and upside pressure is fading. The next likely path is:
    - a cooling phase / pullback toward the nearest support or EMA, OR
    - a choppy range if the higher-timeframe condition is still strongly bullish.
- You MUST NOT flip the overall Bullish/Bearish/Range condition based on divergence alone.
- Do NOT invent new divergence labels or strengths beyond what appears in the verified data.

{rs_clause}
- When describing RSI, your wording MUST match its zone:
- RSI above 80 = overbought / stretched / overheated.
- RSI below 20 = oversold / washed out.
- RSI between 20 and 30 = very weak / near‑oversold.
- RSI between 40 and 60 = mid-range / neutral.
- RSI between 30–40 or 60–70 = transition / leaning weak or strong, not extreme.
You MUST NOT call an RSI above 70 "middle of the range" or "neutral".

- Clarify which signals are strong vs noisy or contradictory.
- Highlight what the signals imply for:
- buyers (longs),
- sellers (shorts, or those exiting longs),
- and those waiting on the sidelines.

- Summarize the **overall bias** clearly (Bullish / Bearish / Range) for the {persona_label} trader, based ONLY on the regimes and structure in the analysis.

============================================================================
GANN RULES – CONFIRMATION ONLY (NEW)
============================================================================

When GANN_METRICS are present in the analysis (look for the GANN_METRICS header), use them ONLY as supporting context:

CRITICAL: GANN signals MUST NOT:
- Override the primary market regimes (Bullish/Bearish/Range/SmartRange/RetailChop)
- Change allowed direction (Long-only / Short-only)
- Modify the trading strategies
- Flip bias against the primary Triple-Screen regime

What you CAN do with GANN signals:
- Upgrade conviction: When GANN aligns with regime (e.g., Bullish regime + Friday Weekly High)
- Downgrade conviction: When GANN contradicts (e.g., Bullish regime + Quarterly Breakdown)
- Add early warnings: Deeper correction, volume spikes, 30 DMA breaks
- Provide additional confluence in narrative: "GANN 4-week breakout confirms the Bullish regime"

Key GANN signals to mention (if present):
1. Friday Weekly High/Low → Next week directional bias
2. 4-Week High/Low Breakout → Trend continuation signal
3. 3-Day High Break → 4th day surge expectation
4. 5:3 / 9:5 Correction Ratios → Pullback duration expectations
5. Volume Spike in Consolidation → Trend change signal
6. Monthly Double/Triple Bottoms/Tops → Long-term accumulation/distribution
7. Quarterly Breakout → Major trend reversal signal
8. 30 DMA Break → Correction warning

Use GANN to enrich the narrative, NOT to override the systematic regime framework.

{lower_tf_indicator_clause}
- You MAY mention numeric indicator values on intraday timeframes (1H, 30m, 15m, 5m), but ONLY if those exact numbers appear in the analysis / verified data. Do NOT invent any new intraday indicator values.
- When you quote intraday indicators (EMA, RSI, MACD, StochRSI, ATR, MFI, etc.), use them as supporting context for structure and bias, not as independent trade triggers.
- If the analysis does not provide a specific intraday number for an indicator, describe that indicator qualitatively instead (for example: "short-term momentum is fading", "volatility is elevated") without making up numbers.

- On intraday timeframes (1H, 30m, 15m, 5m), you MUST NOT invent any new numeric levels for supply/demand zones or intraday support/resistance.
- When you mention intraday supply or demand, reuse only the existing levels and zones that appear in the analysis (for example, prior swing highs/lows, PDH/PDL, or PRECOMPUTED structure on those timeframes).
- Do NOT create fresh intraday prices like "supply near 316.2 and demand near 314.0" unless those exact numbers already exist in the analysis for that timeframe.
- Avoid placing supply and demand at almost the same price (for example, 313 and 314). If you mention both, they should clearly represent different areas, not overlapping ticks.

- Encourage mentioning key numbers for Daily (e.g., "Daily EMA50 near X", "Daily RSI around Y").
- De‑prioritize bands/channels and volume for speech‑style explanation unless they truly matter.
- Avoid repeating the same price level more than 2 times within any single section. If you have already used a level (for example, a support around X) in that section, refer to different levels, indicators, or structures when you need more detail.
- In sections 1–3, mention at least:
- one EMA or RSI value from the higher timeframes (Weekly, Daily, Monthly or Quarterly), and
- one other higher-timeframe concept such as volatility via ATR14, overall range width, or a clear market condition description (for example, "well-defined range" or "mixed environment"),
so the explanation does not rely only on support/resistance levels.

- You are also given a Market Stage label for the Trend timeframe (Accumulation, Advancing, Distribution, or Declining) inside the analysis text. You MUST:
- Explicitly mention this Market Stage in the big-picture section.
- Explain how this stage fits with the higher timeframe bias (for example, Bullish & Advancing, Bearish & Declining, Range & Distribution, SmartRange & Accumulation).
- Align your recommended trade types (trend-following, pullback, range-edge sell near resistance or buy near support, liquidity sweep reversal, mean-reversion) with BOTH the higher timeframe bias AND the Market Stage.

- For Swing mode, when describing the Daily timeframe:
- If Daily RSI is above 80 OR the Daily close is above the Daily upper Bollinger Band (3 standard deviations), you MUST explicitly call the Daily overbought / stretched / exhausted and link this to mean‑reversion or profit‑protection ideas (for example, "selling bounces off resistance" or "taking profits on stretched longs"), rather than new aggressive trend‑following longs.

- If the analysis mentions that the Trend timeframe close is ABOVE the upper Bollinger Band (3 standard deviations), you MUST treat that as an extreme overbought / exhaustion area:
- Explicitly call this out as a place to take profits, avoid new aggressive longs, or consider cautious mean‑reversion ideas ONLY when this does not violate the higher‑timeframe bias and Triple‑Screen rules.

- When the Trend timeframe (the first in {tf_explainer}) close is ABOVE the upper Bollinger Band (3 standard deviations), you MUST say that price is currently trading above the upper band and tell the trader to watch for failure to sustain above and reversion back inside the bands. Do NOT say "holds below the upper band" in this situation.

- Only use language like "holds below the upper Bollinger Band" when price is currently inside the bands.

- If the analysis mentions that the Trend timeframe close is BELOW the lower Bollinger Band (3 standard deviations), you MUST treat that as an extreme oversold / capitulation area:
- Explicitly call this out as a place to look for high‑quality long mean‑reversion or pullback entries, or to avoid new aggressive shorts, again only when consistent with the higher‑timeframe bias and Triple‑Screen rules.

- When you include FO_METRICS in your explanation, use wording like:
- "ATM Call IV is around X and falls in the <low/normal/high> band."
- "ATM Put IV is around Y and falls in the <low/normal/high> band."
- "Both sides are in a normal IV zone" or "Put IV is higher than call IV, indicating demand for downside protection."
- "Volatility shows put_skew (puts are more expensive), suggesting traders are hedging against downside."
- "Net delta bias is bullish, confirming institutional buying pressure."
- "Gamma exposure is high, so expect quick moves and consider tighter stops."
- "Liquidity is good, so execution should be clean."

- Whenever you see words like "bullish skew" or "bearish skew" in the analysis, rewrite them in simple language as "options positioning is tilted bullish/bearish (more puts than calls / more calls than puts), so traders are leaning to the upside/downside", and avoid using the word "skew" itself.

Style and structure:
- Sound like a clear, engaging TV stock‑market presenter explaining the stock to active traders in a {persona_label} style: confident, simple language, minimal jargon.
- Write for beginner‑to‑intermediate traders: explain terms simply (for example, instead of "micro BOS/CHOCH", say "small trend changes or reversals").
- When mentioning "key levels", "support", or "resistance" on any timeframe, use specific numeric price levels or zones taken from the analysis where it genuinely helps. For higher timeframes (Weekly, Daily, Monthly, Quarterly), this is strongly encouraged. For intraday timeframes (1H, 30m, 15m, 5m), you MAY also use numeric levels and indicator values, but ONLY if those exact numbers already appear in the analysis / verified data.
- Make the language more engaging and energetic, but still professional.
- Use short paragraphs and bullet points where helpful so the trader can scan quickly.
- Avoid deep technical jargon (like "MACD histogram divergence" or "micro BOS/CHOCH"). Use plain phrases like "momentum is cooling off" or "price is stuck in a sideways band".

- For the Swing persona specifically, focus on trades that can run for several days to a few weeks. Prioritise asymmetric entries from clear range edges or strong structural levels with room to mean‑revert, not intraday scalps.

Required sections (do NOT skip):

1) Big-picture bias and market condition
- In 3–4 sentences, explain what the higher timeframe(s) in {tf_explainer} are saying about the overall bias (Bullish / Bearish / Range / choppy).
- State clearly whether trend-following trades are ON or OFF for this {persona_label} context right now.
- When the Trend timeframe is trending (Bullish or Bearish) but the Setup and/or Entry timeframes are in a Range or SmartRange market condition, you MUST say that fresh trend-following trades are OFF and that only pullback / range-edge / mean-reversion ideas are appropriate until lower timeframes realign.
- Explicitly mention the Market Stage label (Accumulation, Advancing, Distribution, or Declining) and link it to likely next behaviour (for example, "Advancing phase in an uptrend usually leads to continuation unless a key support level like <exact support from the analysis> fails").
- You are given the exact Market Stage for the Trend timeframe as MARKET_STAGE_TREND_TF in the context above (for example, "MARKET_STAGE_TREND_TF: Accumulation"). When the Trend timeframe is tagged with a specific Market Stage in the analysis (Accumulation, Advancing, Distribution, or Declining), you MUST:
    - Quote that label EXACTLY in your explanation, for example: "The Trend timeframe is tagged in the analysis as being in an Accumulation / Advancing / Distribution / Declining phase (shown above as MARKET_STAGE_TREND_TF)...".
    - You MUST NOT substitute a different word. If the context says "MARKET_STAGE_TREND_TF: Accumulation", you MUST write the word "Accumulation" in your sentence (not "Declining" or any other label). Likewise, if it says "Advancing", you MUST literally write "Advancing"; if it says "Distribution", you MUST literally write "Distribution"; if it says "Declining", you MUST literally write "Declining".
    - You may adapt the surrounding sentence pattern, but you MUST NOT invent any Market Stage label that is not exactly one of the provided values.
- When the Trend timeframe is tagged as a choppy backdrop (for example, RetailChop or a weak range) but lower timeframes clearly align in one direction, you MUST highlight that the backdrop is noisy and that any directional trades should be smaller, more conservative, and taken only from strong PRECOMPUTED structure (for example, selling bounces near resistance or buying dips near support).
- Include at least one sentence that forecasts the most likely path over the next few candles on the Trend timeframe (continuation, pullback, or sideways pause), consistent with the regimes, MARKET_STAGE_TREND_TF, and structure.
- Wherever it genuinely helps, anchor the big-picture view to one or two higher-timeframe indicator or structure levels from the analysis (for example, "downward pressure toward the lower Weekly Keltner band near <exact level>" or "room to retest the recent Weekly Darvas lower edge around <exact level>").
- When the Trend timeframe is choppy but both Setup and Entry timeframes clearly lean Bearish, you should use language like "short-favoured, look to sell pullbacks into resistance" rather than calling for aggressive longs; if they clearly lean Bullish, use "long-favoured, look to buy pullbacks into support" instead.
{big_picture_fo_extra}

2) Multi-timeframe read ({tf_explainer})
- Briefly describe how the Trend, Setup, and Entry timeframes line up (trend, range, compression, key zones).
- Mention only the most important indicators and structures where they add real context.
- For the higher timeframes (Weekly, Daily, Monthly or Quarterly), call out 1–2 specific numeric price levels taken from the analysis for each – especially Daily support, resistance, Darvas box upper/lower/mid, and key EMAs like Daily EMA50 or EMA200 – without inventing any new prices.
- On intraday timeframes (1H, 30m, 15m, 5m), you MAY mention numeric indicator values and intraday price levels, but ONLY if those exact numbers already appear in the analysis / verified data. Do NOT invent new intraday prices or indicator values.
- For each timeframe, state what is MORE likely to happen next (continue trend, retest support, retest resistance, or chop in a band) and what level would invalidate that expectation.
- If a Darvas box is present on the Trend timeframe, you MUST say whether price is currently inside, above, or below that box and what that implies (for example, trading below the recent Daily price box means rallies back toward the old floor are likely to attract sellers).

3) Concrete trading stance for this persona
- Clearly answer: "Given this context, what types of trades should a {persona_label} trader focus on now, and which should they avoid?"
- Explain in simple terms: instead of technical jargon, say things like "buying pullbacks to support" or "selling bounces off resistance".

- If (and ONLY if) the persona is F&O (FO / FNO) AND FO_METRICS / OPTIONS & FUTURES METRICS are present in the analysis, you MUST add 2-3 sentences explaining:
    * How PCR/OI, volatility skew, and net delta bias affect the directional view.
    * How gamma exposure and liquidity should influence position sizing and stop placement.
    * Whether the risk profile suggests aggressive, moderate, or conservative approach.
- For F&O personas only, if options positioning is against the Daily bias (for example, low PCR in a bullish trend), you MUST recommend more selective entries, tighter stops, or smaller size, even if direction remains unchanged.

- For the Positional persona, when you describe short-side ideas, you MUST frame them primarily as risk-management or profit-booking on existing long positions (for example, "taking partial profits into strength" or "hedging or trimming near resistance"), not as aggressive new standalone short trades. New short entries for Positional are allowed only when the higher timeframes are clearly Bearish and the filters explicitly permit SHORT, and even then they should be presented as cautious, tactical ideas with smaller size.
- Be explicit about allowed vs discouraged directions WITH specific numeric price examples taken from the analysis (for example, "buy pullbacks near support around 316" or "avoid new longs if price stalls below resistance near 321.5").
- For Intraday and F&O personas, when the big-picture bias is bearish but lower timeframes are mixed, you MUST say that traders should only look for tactical shorts into clearly identified resistance levels from the analysis and avoid chasing breakdowns at new lows.
- When the Trend timeframe is choppy (sideways / mixed) but both the Setup and Entry timeframes are clearly Bearish, you SHOULD state that the environment is short‑leaning overall and that any longs from support are tactical mean‑reversion only, not the primary trend idea.
- For each allowed idea, give a simple IF–THEN template that includes the actual level, for example: "If price approaches the resistance level around 321.5 from the analysis and fails to break, sellers can look for short entries with stops just above that level."

- Across sections 1), 2), and 3), you MUST mention at least:
- one EMA or RSI value from the higher timeframes (Weekly, Daily, Monthly or Quarterly), and
- one other higher-timeframe concept such as volatility via ATR14, overall range width, a clear market condition description (for example, "well-defined range" or "mixed environment"), or Relative Strength (RS) vs the index when RS is present in the verified data,
so the explanation does not rely only on support/resistance levels.

4) Timeframe-specific action plan (MOST IMPORTANT)
- Provide a short, bullet-point checklist for each stage in {tf_explainer}:

    - **Trend timeframe** (first in {tf_explainer}): what to watch, including 1–2 specific numeric price levels that matter taken from the analysis (for example, "watch if it holds above the nearest support level around 316.0 or breaks below the recent swing low near 307.5").
    - Include at least one forecast bullet: "Most likely, price will do X as long as it stays above/below <specific level from the analysis>."
    - Include one invalidation bullet: "If instead price does Z at <specific level from the analysis>, the current view is invalid and you should stand aside or re-evaluate."

    - **Setup timeframe** (middle in {tf_explainer}): what price behaviour to expect near key levels, again using actual levels from the analysis (for example, "look for buying pressure around the identified support zone near 314.0" or "watch for rejection near resistance around 321.5").

    - **Entry timeframe** (last in {tf_explainer}): describe the type of price action to wait for (for example, "wait for a clear bounce off the existing support zone with buyers stepping in"), but do NOT invent any new price levels or intraday indicator numbers.
    - You MAY mention numeric intraday price levels and indicator values (1H, 30m, 15m, 5m) at this stage, but ONLY if those exact numbers already appear in the analysis / verified data.

- Every bullet must reference specific numeric levels or PRECOMPUTED structures already present in the analysis (support/resistance, Darvas box levels, Fib levels, prior swing highs/lows, PDH/PDL, or intraday structure levels from the engine).
- Include at least one clear "no-trade" condition in the checklist (for example, "Stand aside if price is stuck in the middle of the range between two key levels from the analysis with no clear rejection.").

5) Market psychology (brief)
- In 4–5 sentences, explain who is currently under pressure or in control (late longs near resistance, shorts trapped after a breakout, range traders caught in whipsaws), using the actual regimes, Darvas/Fib structure, support–resistance, and RSI/volatility context from the analysis.
- When FO_METRICS / OPTIONS & FUTURES METRICS are present, include 1-2 sentences on:
    * How options positioning (PCR/OI, volatility skew) reflects crowd behaviour (fear/greed)
    * What gamma exposure implies about potential whipsaws
    * Whether futures OI state reinforces or contradicts the price trend
- Where RSI divergence is present on any higher timeframe, briefly state which side (buyers or sellers) is losing momentum and what that implies for the next phase (continuation, cooling off, or mean‑reversion toward a specific level from the analysis).
- Avoid metaphors; use clear trading language that directly links psychology to concrete scenarios like breakout follow‑through, failed breakouts, trend exhaustion, or range rejections around the given levels.

6) What to watch next
- Give a concise, bullet‑style checklist for the next few bars on each relevant timeframe in {tf_explainer} (Trend, Setup, Entry), focusing only on the most important levels and signals from the analysis.
- For the higher timeframes (Weekly, Daily, Monthly or Quarterly), mention 2–3 specific numeric price levels or zones from the analysis to monitor (for example, a key support, an important resistance, a Darvas box edge, or a Fib band), without inventing any new prices. For intraday timeframes (1H, 30m, 15m, 5m), you MAY also mention numeric price levels and indicator values, but ONLY if those exact numbers already appear in the analysis / verified data.
- In the same section, highlight 1–2 indicator or market‑condition signals that matter for the next moves on the relevant timeframes. For higher timeframes (Weekly, Daily, Monthly, Quarterly), this can include items like Weekly EMA50 still below price, Daily RSI near overbought, or ATR14 showing expanding volatility. For intraday timeframes (1H, 30m, 15m, 5m), you MAY also mention numeric indicator values (EMA, RSI, ATR, VWAP, etc.), but ONLY if those exact numbers already appear in the analysis / verified data, and use them to support the trade plan rather than invent new triggers.
- For each level or signal, explain briefly how it would change or confirm the trading stance, using forward‑looking language like "If this happens, the next likely step is…" or "If this fails at <specific level>, expect…". Keep all intraday references (1H, 30m, 15m, 5m) tied strictly to the actual intraday price levels and indicator values already present in the analysis, and do NOT invent any new intraday numbers.
- If (and ONLY if) the persona is F&O (FO / FNO) AND FO_METRICS / OPTIONS & FUTURES METRICS are present in the analysis, include 1–2 watch items related to the options market (for example, "watch if put skew intensifies" or "monitor gamma buildup near key strikes").

Hard constraints:
- Do NOT generate any new price levels, indicators, or structures. Only refer to values, zones, and concepts already present in the analysis.
- Do NOT change the direction of any strategy or regime vs what is implied in the analysis; you may criticise weak strategies, but you cannot flip them or invent new ones.
- Focus on concepts, structure, and trade logic — not on minor numeric fluctuations.
- Every mention of "support", "resistance", "key level", or structure on ANY timeframe MUST use specific numeric prices or zones taken from the analysis (for example, "support near 2450" or "resistance around 2510") and MUST NOT invent new levels. For intraday timeframes (1H, 30m, 15m, 5m), you MAY mention numeric price levels and indicator values, but ONLY if those exact numbers already appear in the analysis / verified data.
- The final explanation must make it very clear what a {persona_label} trader should and should NOT attempt on the Trend, Setup, and Entry timeframes under the current conditions.
"""  # <-- THIS CLOSES THE f""" THAT STARTED THE LONG PROMPT

    # ========== COMPREHENSIVE MANDATORY OUTPUT RULES ==========
    mandatory_rules = """
**╔═══════════════════════════════════════════════════════════════════════════════╗**
**║                    ⚠️ MANDATORY OUTPUT REQUIREMENTS ⚠️                         ║**
**║         YOU MUST INCLUDE ALL OF THE FOLLOWING IN YOUR NARRATIVE               ║**
**╚═══════════════════════════════════════════════════════════════════════════════╝**

Based on the ADDITIONAL VERIFIED DATA section above, you MUST include these in your trading plan:

**1. RSI DIVERGENCE (MANDATORY - IF PRESENT IN DATA):**
   - List each timeframe with divergence (30M, 4H, Daily, Weekly, etc.)
   - State type (bullish/bearish) and strength value
   → Example: "30-minute chart shows BEARISH divergence (strength: 1.29)"
   → Example: "4-hour chart shows BULLISH divergence (strength: 1.23)"
   → Example: "Weekly chart shows BEARISH divergence (strength: 1.24)"

**2. RELATIVE STRENGTH - MANSFIELD (MANDATORY - IF PRESENT IN DATA):**
   - State Daily RS bucket and Mansfield value
   - State Weekly RS bucket and Mansfield value
   → Example: "Daily RS: Neutral (Mansfield: +0.073)"
   → Example: "Weekly RS: Neutral (Mansfield: +0.098)"

**3. BOLLINGER BANDS (MANDATORY - IF PRESENT IN DATA):**
   - State position: above upper (overbought), below lower (oversold), or inside bands
   - Include percentage if inside
   → Example: "Price is inside Bollinger Bands (65% from lower to upper)"
   → Example: "Price is ABOVE upper Bollinger Band → OVERBOUGHT"

**4. FIBONACCI LEVELS (MANDATORY - IF PRESENT IN DATA):**
   - List the key levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
   - State current price position relative to these levels
   → Example: "Fibonacci levels: 23.6% at 23500, 38.2% at 24000, 50% at 24500"
   → Example: "Price is pulling back into the 38.2-61.8% retracement zone"

**5. GANN SIGNALS (MANDATORY - IF PRESENT IN DATA):**
   - Friday weekly high/low signal with confidence percentage
   - Monthly double/triple top/bottom patterns with gap months
   - 3-day high break signal (if present)
   - 4-week high/low break signal (if present)
   - Quarterly breakout signal (if present)
   → Example: "GANN: Friday made weekly high (65% confidence) → bullish bias next week"
   → Example: "GANN: Monthly double top detected (14 months gap) → bearish signal"
   → Example: "GANN: 3-day high broken → expect 4th day surge"

**6. DARVAS BOX (MANDATORY - IF PRESENT AND WITHIN PROXIMITY):**
   - State upper, lower, mid levels
   - State current position (inside/above_upper/below_lower)
   - State strength score (0-10)
   → Example: "Darvas Box: Upper 26373, Lower 22182, Mid 24278"
   → Example: "Price is INSIDE the box (strength: 4.5/10)"

**7. KELTNER CHANNELS (IF PRESENT):**
   - State if KC_tight (compression) is True or False
   → Example: "Keltner Channels show compression (KC_tight=True), suggesting a breakout may be near"

**8. ORDER BLOCKS & LIQUIDITY (IF PRESENT IN DATA):**
   - Mention any bullish/bearish order blocks near current price
   - Mention any liquidity pools (EQH/EQL) if present
   → Example: "Bearish order block detected between 24025-24037"
   → Example: "Liquidity pool at recent highs (24074) - potential stop hunt zone"

**FAILURE TO INCLUDE THESE WILL MAKE YOUR ANALYSIS INCOMPLETE.**

Here is the analysis:

"""
    
    trainer_prompt += mandatory_rules
    trainer_prompt += clean_output

    # ========== CALL GEMINI (Primary) ==========
    GENAI_API_KEY = os.environ.get("GENAI_API_KEY") or ""
    
    if not GENAI_API_KEY:
        raise ValueError("GENAI_API_KEY not set for trainer explanation")
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GENAI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        response = model.generate_content(
            trainer_prompt,
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 3000,
                "top_p": 0.95,
            }
        )
        
        # ========== TOKEN TRACKING ==========
        try:
            usage_metadata = response.usage_metadata
            if usage_metadata:
                prompt_tokens = usage_metadata.prompt_token_count
                candidates_tokens = usage_metadata.candidates_token_count
                total_tokens = usage_metadata.total_token_count
                
                print(f"[TOKEN USAGE - Trainer Gemini]")
                print(f"  Prompt tokens: {prompt_tokens}")
                print(f"  Response tokens: {candidates_tokens}")
                print(f"  Total tokens: {total_tokens}")
                
                # Optional: Store in session state for UI display
                st.session_state.token_usage = {
                    "prompt": prompt_tokens,
                    "response": candidates_tokens,
                    "total": total_tokens
                }
            else:
                print("[TOKEN USAGE - Trainer Gemini] No usage metadata available")
        except AttributeError:
            print("[TOKEN USAGE - Trainer Gemini] Usage metadata not supported")
        except Exception as e:
            print(f"[TOKEN USAGE - Trainer Gemini] Error getting token count: {e}")
        # ==================================
        
        return response.text
    except Exception as e:
        print(f"ERROR: Gemini trainer explanation failed: {e}")
        print("Falling back to template-based explanation using precomputed data...")
        
        # Get current price based on persona (NOT hardcoded to Daily)
        try:
            if persona_key in ("positional", "position"):
                current_price = precomputed.get("MONTHLY", {}).get("indicators", {}).get("MN_Close", 0)
            elif persona_key in ("investing", "investment"):
                current_price = precomputed.get("QUARTERLY", {}).get("indicators", {}).get("Q_Close", 0)
            elif persona_key == "swing":
                current_price = precomputed.get("WEEKLY", {}).get("indicators", {}).get("W_Close", 0)
            else:
                current_price = precomputed.get("DAILY", {}).get("indicators", {}).get("D_Close", 0)
        except Exception as price_err:
            print(f"DEBUG: Could not extract current_price: {price_err}")
            current_price = 0
        
        # Get regimes (already available)
        regimes_local = regimes if 'regimes' in locals() else {}
        
        # Get strategies (if available)
        strategies_local = {}
        try:
            if 'final_strategies' in locals():
                strategies_local = final_strategies
        except:
            pass
        
        # Call persona-aware fallback
        return generate_fallback_trainer_explanation(
            precomputed=precomputed,
            regimes=regimes_local,
            strategies=strategies_local,
            current_price=current_price,
            supports=[],
            resistances=[],
            market_stage=market_stage
        )

def extract_balanced_json(text):
    """Extract first valid JSON object/array from arbitrary text.

    - Ignores braces/brackets that appear inside JSON strings.
    - Handles escaped quotes and escaped backslashes correctly.
    - Scans all candidate '{' or '[' positions until a decodable JSON is found.
    """
    if not text:
        return None

    # Remove markdown code fences (```json ... ```)
    cleaned = re.sub(r'```json\s*|\s*```', '', text, flags=re.IGNORECASE)

    # Collect all potential JSON start positions
    start_positions = [i for i, ch in enumerate(cleaned) if ch in "{["]

    open_chars = {"{": "}", "[": "]"}
    close_chars = {"}": "{", "]": "["}

    for start_idx in start_positions:
        stack = []
        json_chars = []
        in_string = False
        escape_next = False

        for i in range(start_idx, len(cleaned)):
            ch = cleaned[i]
            json_chars.append(ch)

            if in_string:
                # Inside a quoted string: track escapes and closing quote only
                if escape_next:
                    escape_next = False
                elif ch == "\\":
                    escape_next = True
                elif ch == '"':
                    in_string = False
            else:
                # Outside string: look for string start or structural brackets
                if ch == '"':
                    in_string = True
                    escape_next = False
                elif ch in open_chars:
                    stack.append(ch)
                elif ch in close_chars:
                    if not stack:
                        # Unbalanced closing bracket → abandon this candidate
                        break
                    last_open = stack.pop()
                    if close_chars[ch] != last_open:
                        # Mismatched brackets → abandon this candidate
                        break
                    if not stack:
                        # Balanced JSON candidate found
                        json_str = "".join(json_chars)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            # Candidate is not valid JSON; try next start position
                            break

    return None

def safe_parse_json(response_text, schema=None):
    """Safely parse JSON using balanced bracket extraction with None handling"""
    try:
        # Debug: Log raw response if parsing fails
        if not response_text or len(response_text) < 10:
            print(f"⚠️ Empty or very short response: {response_text}")
            return None
        
        cleaned = re.sub(r'```json\s*|\s*```', '', response_text)
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)  # Remove control chars
        parsed = extract_balanced_json(cleaned)
        
        # Debug: Log what we parsed
        if parsed and isinstance(parsed, dict):
            print(f"✅ Parsed JSON with keys: {list(parsed.keys())}")
            if "futures" in parsed:
                print(f"   futures.oi_state: {parsed['futures'].get('oi_state')}")
            if "sentiment" in parsed:
                print(f"   sentiment: {parsed.get('sentiment')}")
        
        if parsed and schema:
            # Deep copy to avoid modifying original
            import copy
            parsed = copy.deepcopy(parsed)
            
            # Convert None values to acceptable defaults before validation
            if isinstance(parsed, dict):
                # Handle futures.oi_state
                if "futures" in parsed and isinstance(parsed["futures"], dict):
                    if parsed["futures"].get("oi_state") is None:
                        parsed["futures"]["oi_state"] = "short_covering"
                        print("   🔧 Converted None oi_state → short_covering")
                    if parsed["futures"].get("conviction") is None:
                        parsed["futures"]["conviction"] = "neutral"
                        print("   🔧 Converted None conviction → neutral")
                
                # Handle gamma_exposure
                if parsed.get("gamma_exposure") is None:
                    parsed["gamma_exposure"] = "moderate"
                    print("   🔧 Converted None gamma_exposure → moderate")
                
                # Handle volume_momentum
                if parsed.get("volume_momentum") is None:
                    parsed["volume_momentum"] = "flat"
                    print("   🔧 Converted None volume_momentum → flat")
                
                # Handle positioning
                if parsed.get("positioning") is None:
                    parsed["positioning"] = "Balanced"
                    print("   🔧 Converted None positioning → Balanced")
                
                # Handle term_structure
                if parsed.get("term_structure") is None:
                    parsed["term_structure"] = "flat"
                    print("   🔧 Converted None term_structure → flat")
                
                # Handle liquidity
                if parsed.get("liquidity") is None:
                    parsed["liquidity"] = "good"
                    print("   🔧 Converted None liquidity → good")
                
                # Handle sentiment (for TREND_SCHEMA compatibility)
                if parsed.get("sentiment") is None:
                    parsed["sentiment"] = "Neutral"
                    print("   🔧 Converted None sentiment → Neutral")
                
                # Handle iv_regime
                if parsed.get("iv_regime") is None:
                    parsed["iv_regime"] = "Normal"
                    print("   🔧 Converted None iv_regime → Normal")
                
                # Handle fo_decision
                if "fo_decision" in parsed and isinstance(parsed["fo_decision"], dict):
                    if parsed["fo_decision"].get("no_trade") is None:
                        parsed["fo_decision"]["no_trade"] = False
                        print("   🔧 Converted None fo_decision.no_trade → False")
            
                # ========== ADD THIS NEW BLOCK HERE ==========
                # Generic: Convert any unexpected string to a safe default
                for key in ["gamma_exposure", "volume_momentum", "liquidity", "term_structure"]:
                    if key in parsed and parsed[key] not in [None, "low", "moderate", "high", "extreme", "rising", "flat", "falling", "poor", "fair", "good", "excellent", "normalcontango", "frontelevated"]:
                        print(f"   🔧 Converted unexpected {key} value '{parsed[key]}' → default")
                        if key == "gamma_exposure":
                            parsed[key] = "moderate"
                        elif key == "volume_momentum":
                            parsed[key] = "flat"
                        elif key == "liquidity":
                            parsed[key] = "good"
                        elif key == "term_structure":
                            parsed[key] = "flat"
                # ========== END OF NEW BLOCK ==========
                
                # Also handle "neutral" for oi_state (add this too)
                if "futures" in parsed and isinstance(parsed["futures"], dict):
                    oi_state = parsed["futures"].get("oi_state")
                    if oi_state == "neutral":
                        parsed["futures"]["oi_state"] = "short_covering"
                        print(f"   🔧 Converted 'neutral' oi_state → short_covering")
            
            # Now validate with schema
            from jsonschema import validate
            validate(instance=parsed, schema=schema)
            print("✅ JSON validation successful")
        
        return parsed
        
    except Exception as e:
        print(f"❌ JSON parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# JSON Schemas for LLM validation
TREND_SCHEMA = {
    "type": "object",
    "properties": {
        "bias": {"type": ["string", "null"], "enum": ["Bullish", "Bearish", "Range", None]},
        "market_stage": {"type": ["string", "null"], "enum": ["Advancing", "Declining", "Accumulation", "Distribution", None]},
        "trend_strength": {"type": ["string", "null"], "enum": ["Strong", "Moderate", "Weak", None]},
        "key_levels": {
            "type": "object",
            "properties": {
                "support": {"type": "array", "items": {"type": "number"}},
                "resistance": {"type": "array", "items": {"type": "number"}}
            }
        },
        "forecast": {"type": ["string", "null"]},
        "regime": {"type": ["string", "null"], "enum": ["Bullish", "Bearish", "Range", "RetailChop", "SmartRange", None]},
        "setup_regime": {"type": ["string", "null"]},
        "entry_regime": {"type": ["string", "null"]},
        "rsi": {"type": ["number", "null"]},
        "adx": {"type": ["number", "null"]},
        "ema_stack": {"type": ["string", "null"], "enum": ["bullish_stack", "bearish_stack", "mixed_stack", None]},
        "darvas_state": {"type": ["string", "null"], "enum": ["inside", "above_upper", "below_lower", None]},
        "darvas_strength": {"type": ["number", "null"]}
    },
    "required": []  # Make all fields optional
}

FO_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": ["string", "null"], "enum": ["Bullish", "Bearish", "Neutral", None]},
        "sentiment_reason": {"type": ["string", "null"]},

        "iv_regime": {"type": ["string", "null"], "enum": ["Low", "Normal", "High", None]},
        "iv_analysis": {
            "type": "object",
            "properties": {
                "call": {"type": ["number", "null"]},
                "put": {"type": ["number", "null"]},
                "alignment": {"type": ["string", "null"], "enum": ["aligned", "skewed", None]},
                "interpretation": {"type": ["string", "null"]}
            }
        },

        "positioning": {
            "type": ["string", "null"],
            "enum": ["Call-heavy", "Put-heavy", "Balanced", None]
        },

        "pcr_analysis": {
            "type": "object",
            "properties": {
                "value": {"type": ["number", "null"]},
                "interpretation": {"type": ["string", "null"]},
                "trend": {"type": ["string", "null"], "enum": ["build_up", "unwinding", "stable", None]}
            }
        },

        "iv_skew": {
            "type": "object",
            "properties": {
                "type": {
                    "type": ["string", "null"],
                    "enum": ["put_skew", "call_skew", "neutral", None]
                },
                "strength": {
                    "type": ["string", "null"],
                    "enum": ["mild", "strong", "extreme", None]
                }
            }
        },

        "gamma_exposure": {
            "type": ["string", "null"],
            "enum": ["low", "moderate", "high", "extreme", None]
        },
        "liquidity": {
            "type": ["string", "null"],
            "enum": ["poor", "fair", "good", "excellent", None]
        },
        "volume_momentum": {
            "type": ["string", "null"],
            "enum": ["rising", "flat", "falling", None]
        },
        "term_structure": {
            "type": ["string", "null"],
            "enum": ["normalcontango", "frontelevated", "flat", None]
        },

        "greeks_summary": {
            "type": "object",
            "properties": {
                "delta_view": {"type": ["string", "null"]},
                "gamma_view": {"type": ["string", "null"]},
                "vega_view": {"type": ["string", "null"]},
                "theta_view": {"type": ["string", "null"]}
            }
        },

        "futures": {
            "type": "object",
            "properties": {
                "oi_state": {
                    "type": ["string", "null"],
                    "enum": [
                        "long_buildup",
                        "short_buildup",
                        "short_covering",
                        "long_unwinding",
                        "neutral",
                        None
                    ]
                },
                "conviction": {
                    "type": ["string", "null"],
                    "enum": ["upgrade", "downgrade", "neutral", None]
                }
            }
        },

        "risk_assessment": {"type": ["string", "null"]},
        "options_advice": {"type": ["string", "null"]},

        "fo_decision": {
            "type": "object",
            "properties": {
                "bias": {
                    "type": ["string", "null"],
                    "enum": ["bullish", "bearish", "range", "no_trade", None]
                },
                "conviction": {
                    "type": ["string", "null"],
                    "enum": ["LOW", "MEDIUM", "HIGH", None]
                },
                "no_trade": {"type": ["boolean", "null"]}
            }
        }
    },
    "required": []  # Make all fields optional
}

# Synthesis Schema for combining trend + FO into narrative
SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "section_1_bias": {"type": "string"},
        "section_2_multitf": {"type": "string"},
        "section_3_stance": {"type": "string"},
        "section_4_action_plan": {"type": "string"},
        "section_5_psychology": {"type": "string"},
        "section_6_watch": {"type": "string"},
        "action_plan": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["Long", "Short", "Wait"]},
                "direction_reason": {"type": "string"},
                "entry_level": {"type": ["number", "null"]},
                "stop_level": {"type": ["number", "null"]},
                "targets": {"type": "array", "items": {"type": "number"}},
                "risk_override": {"type": "string", "enum": ["cautious", "normal", "blocked"]}
            }
        },
        "final_verdict": {"type": "string"}
    },
    "required": []  # Make all fields optional to handle partial responses
}

def build_trend_context_numeric(precomputed, regimes):
    """Build numeric trend context from your actual data structure"""
    daily = precomputed.get("DAILY", {}).get("indicators", {})
    weekly = precomputed.get("WEEKLY", {}).get("indicators", {})
    
    # Extract Darvas info from precomputed if available
    darvas_data = {}
    for tf in ["DAILY", "WEEKLY", "MONTHLY"]:
        tf_data = precomputed.get(tf, {})
        if "darvas_box" in tf_data:
            darvas_data[tf.lower()] = tf_data["darvas_box"]
    
    return {
        "price": {
            "close": daily.get("D_Close"),
            "high": daily.get("High"),
            "low": daily.get("Low"),
            "open": daily.get("Open")
        },
        "regimes": {
            "trend": regimes.get('Trend_Regime', 'Unknown'),
            "setup": regimes.get('Setup_Regime', 'Unknown'),
            "entry": regimes.get('Entry_Regime', 'Unknown')
        },
        "indicators": {
            "rsi": daily.get('D_RSI14'),
            "adx": daily.get('D_ADX14'),
            "di_plus": daily.get('D_DI_PLUS'),
            "di_minus": daily.get('D_DI_MINUS'),
            "atr": daily.get('D_ATR14'),
            "macd": daily.get('D_MACD'),
            "macd_signal": daily.get('D_MACD_signal'),
            "macd_hist": daily.get('D_MACD_hist')
        },
        "emas": {
            "ema10": daily.get('D_EMA10'),
            "ema20": daily.get('D_EMA20'),
            "ema50": daily.get('D_EMA50'),
            "ema200": daily.get('D_EMA200'),
            "stack": daily.get('D_EMA_stack', 'mixed_stack')
        },
        "bollinger": {
            "mid": daily.get('D_BB_mid'),
            "upper": daily.get('D_BB_hi'),
            "lower": daily.get('D_BB_lo')
        },
        "market_stage": regimes.get('Market_Stage', 'Accumulation'),
        "range_score": daily.get('D_RangeScore', 0),
        "darvas": darvas_data
    }

def build_fo_context_numeric(precomputed, fo_context):
    """Build numeric FO context from your actual data structure"""
    fo_metrics = precomputed.get("FO_METRICS", {})
    front = fo_metrics.get("front", {})
    next_expiry = fo_metrics.get("next", {})
    fo_decision = precomputed.get("FO_DECISION", {})
    fo_signals = front.get("fo_signals", {})
    
    return {
        "iv": {
            "call": fo_context.get('iv_call', front.get('atm_iv_call')),
            "put": fo_context.get('iv_put', front.get('atm_iv_put')),
            "call_change": front.get('atm_iv_call_change'),
            "put_change": front.get('atm_iv_put_change'),
            "regime": front.get('iv_regime', 'high'),
            "skew": front.get('iv_skew_atm')
        },
        "pcr": {
            "value": fo_context.get('pcr', front.get('pcr_oi')),
            "change": front.get('pcr_oi_change'),
            "trend": front.get('oi_trend', 'stable')
        },
        "greeks": {
            "call": {
                "delta": front.get('atm_ce_delta'),
                "gamma": front.get('atm_ce_gamma'),
                "theta": front.get('atm_ce_theta'),
                "vega": front.get('atm_ce_vega')
            },
            "put": {
                "delta": front.get('atm_pe_delta'),
                "gamma": front.get('atm_pe_gamma'),
                "theta": front.get('atm_pe_theta'),
                "vega": front.get('atm_pe_vega')
            }
        },
        "oi": {
            "call": {
                "total": front.get('total_call_oi'),
                "change": front.get('total_call_oi_change'),
                "volume": front.get('total_call_volume')
            },
            "put": {
                "total": front.get('total_put_oi'),
                "change": front.get('total_put_oi_change'),
                "volume": front.get('total_put_volume')
            }
        },
        "next_expiry": {
            "iv_call": next_expiry.get('atm_iv_call'),
            "iv_put": next_expiry.get('atm_iv_put'),
            "pcr": next_expiry.get('pcr_oi'),
            "oi_trend": next_expiry.get('oi_trend')
        },
        "signals": {
            "delta_bias": fo_signals.get('delta_bias', fo_context.get('delta_bias')),
            "net_delta": fo_signals.get('net_delta'),
            "gamma_exposure": fo_signals.get('gamma_exposure'),
            "skew_type": fo_signals.get('skew_type'),
            "skew_strength": fo_signals.get('skew_strength'),
            "liquidity": fo_signals.get('liquidity_grade'),
            "volume_momentum": fo_signals.get('volume_momentum'),
            "thin_strikes": fo_signals.get('thin_strikes', []),
            "deep_strikes": fo_signals.get('deep_strikes', [])
        },
        "term_structure": fo_metrics.get('term_structure'),
        "futures": {
            "price": fo_context.get('futures_price'),
            "change": fo_context.get('futures_price_change'),
            "oi_state": fo_context.get('futures_state'),
            "basis": fo_context.get('futures_basis')
        },
        "decision": {
            "bias": fo_decision.get('fo_bias'),
            "conviction": fo_decision.get('fo_conviction'),
            "option_style": fo_decision.get('fo_option_style'),
            "risk_profile": fo_decision.get('fo_risk_profile'),
            "no_trade": fo_decision.get('fo_no_trade', False)
        }
    }

def build_strategy_context(strategies, current_price):
    """Build compact strategy context from Python-authoritative levels"""
    strat_a = strategies.get("A", {})
    strat_b = strategies.get("B", {})
    
    return {
        "current_price": current_price,
        "strategy_a": {
            "type": strat_a.get("type", "Pullback"),
            "entry": strat_a.get("entry", "N/A"),
            "stop": strat_a.get("stop", "N/A"),
            "target1": strat_a.get("target1", "N/A"),
            "target2": strat_a.get("target2", "N/A"),
            "conviction": strat_a.get("conviction", "N/A")
        },
        "strategy_b": {
            "type": strat_b.get("type", "Range-Edge Fade"),
            "entry": strat_b.get("entry", "N/A"),
            "stop": strat_b.get("stop", "N/A"),
            "target1": strat_b.get("target1", "N/A"),
            "target2": strat_b.get("target2", "N/A"),
            "conviction": strat_b.get("conviction", "N/A")
        }
    }

def semantic_validate_trend(trend_data, precomputed):
    """Business-rule validation after schema validation"""
    errors = []
    
    daily = precomputed.get("DAILY", {}).get("indicators", {})
    current_price = daily.get("D_Close", 0)
    
    supports = trend_data.get("key_levels", {}).get("support", [])
    for s in supports:
        if s >= current_price:
            errors.append(f"Support {s} above current price {current_price}")
    
    resistances = trend_data.get("key_levels", {}).get("resistance", [])
    for r in resistances:
        if r <= current_price:
            errors.append(f"Resistance {r} below current price {current_price}")
    
    bias = trend_data.get("bias", "")
    stage = trend_data.get("market_stage", "")
    
    valid_combinations = [
        ("Bullish", "Advancing"), ("Bullish", "Accumulation"),
        ("Bearish", "Declining"), ("Bearish", "Distribution"),
        ("Range", "Accumulation"), ("Range", "Distribution")
    ]
    
    if (bias, stage) not in valid_combinations:
        errors.append(f"Invalid bias/stage combo: {bias}/{stage}")
    
    rsi = daily.get("D_RSI14", 50)
    if bias == "Bullish" and rsi < 30:
        errors.append(f"Bullish bias but RSI oversold at {rsi}")
    if bias == "Bearish" and rsi > 70:
        errors.append(f"Bearish bias but RSI overbought at {rsi}")
    
    return errors

def semantic_validate_fo(fo_data, fo_context):
    """Validate FO business rules with null-safe checks"""
    errors = []
    
    # Get PCR from fo_context (not fo_data) - with null safety
    pcr = fo_context.get('pcr', 1.0)
    
    # NULL-SAFE: Convert None to a default value
    if pcr is None:
        pcr = 1.0  # Default neutral value
        print("⚠️ semantic_validate_fo: PCR was None, defaulting to 1.0")
    
    positioning = fo_data.get("positioning", "")
    
    # Only validate if we have valid numbers
    if pcr is not None and positioning:
        try:
            pcr_float = float(pcr)
            if pcr_float < 0.8 and positioning == "Put-heavy":
                errors.append(f"PCR {pcr_float} < 0.8 indicates call-heavy, not put-heavy")
            if pcr_float > 1.2 and positioning == "Call-heavy":
                errors.append(f"PCR {pcr_float} > 1.2 indicates put-heavy, not call-heavy")
            if 0.8 <= pcr_float <= 1.2 and positioning not in ["Balanced", "Neutral"]:
                errors.append(f"PCR {pcr_float} is balanced but positioning is {positioning}")
        except (TypeError, ValueError):
            errors.append(f"PCR value '{pcr}' is not a valid number")
    
    # Get IV from fo_context with null safety
    call_iv = fo_context.get('iv_call', 20)
    put_iv = fo_context.get('iv_put', 20)
    
    # Convert None to defaults
    if call_iv is None:
        call_iv = 20
        print("⚠️ semantic_validate_fo: call_iv was None, defaulting to 20")
    if put_iv is None:
        put_iv = 20
        print("⚠️ semantic_validate_fo: put_iv was None, defaulting to 20")
    
    iv_regime = fo_data.get("iv_regime", "")
    
    # Only validate if we have valid numbers
    try:
        call_iv_float = float(call_iv)
        put_iv_float = float(put_iv)
        
        if iv_regime == "Low" and (call_iv_float > 25 or put_iv_float > 25):
            errors.append(f"IV regime Low but IVs are {call_iv_float}/{put_iv_float}")
        if iv_regime == "High" and (call_iv_float < 20 or put_iv_float < 20):
            errors.append(f"IV regime High but IVs are {call_iv_float}/{put_iv_float}")
    except (TypeError, ValueError):
        errors.append(f"IV values are invalid: call_iv={call_iv}, put_iv={put_iv}")
    
    # Check FO Decision consistency if available (with null safety)
    fo_decision = fo_data.get("fo_decision", {})
    if fo_decision and fo_decision.get("no_trade") and fo_data.get("sentiment") != "Neutral":
        errors.append(f"no_trade=True but sentiment is {fo_data.get('sentiment')}")
    
    return errors

class ActionType:
    LONG = "Long"
    SHORT = "Short"
    WAIT = "Wait"

def merge_with_priority(python_strategies, fo_decision, trend_data, fo_data, synthesis_data):
    """Deterministic merge with Python authority at top and null-safe fallbacks."""

    def as_dict(value):
        return value if isinstance(value, dict) else {}

    def has_meaningful_dict(value):
        return isinstance(value, dict) and len(value) > 0

    def safe_float(value, default=None):
        try:
            if value in (None, "", "N/A"):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def text(value, default=""):
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip() or default
        return str(value)

    def infer_action_from_strategy(strategy, trend):
        strategy = as_dict(strategy)
        trend = as_dict(trend)

        entry_val = safe_float(strategy.get("entry"))
        if entry_val is None or entry_val <= 0:
            return ActionType.WAIT, "Wait"

        strategy_type = text(strategy.get("type"), "").lower()
        trend_bias = text(trend.get("bias"), "Neutral").lower()

        if "short" in strategy_type:
            return ActionType.SHORT, ActionType.SHORT
        if "long" in strategy_type:
            return ActionType.LONG, ActionType.LONG

        if "pullback" in strategy_type:
            if trend_bias == "bullish":
                return ActionType.LONG, ActionType.LONG
            if trend_bias == "bearish":
                return ActionType.SHORT, ActionType.SHORT

        if "fade" in strategy_type:
            if trend_bias == "bearish":
                return ActionType.SHORT, ActionType.SHORT
            if trend_bias == "bullish":
                return ActionType.LONG, ActionType.LONG

        return ActionType.WAIT, "Wait"

    python_strategies = as_dict(python_strategies)
    fo_decision = as_dict(fo_decision)
    trend_data = as_dict(trend_data)
    fo_data = as_dict(fo_data)
    synthesis_data = as_dict(synthesis_data)

    strat_a = as_dict(python_strategies.get("A"))
    strat_b = as_dict(python_strategies.get("B"))

    trend_available = has_meaningful_dict(trend_data)
    fo_available = has_meaningful_dict(fo_data)
    synthesis_available = has_meaningful_dict(synthesis_data)

    llm_count = sum([trend_available, fo_available, synthesis_available])
    if llm_count == 3:
        status = "ok"
    elif llm_count >= 1:
        status = "partial"
    else:
        status = "fallback"

    # Priority 1: Python hard risk rules
    if fo_decision.get("fo_no_trade", False):
        return {
            "status": status if status != "fallback" else "partial",
            "final_action": ActionType.WAIT,
            "final_action_label": "Wait",
            "final_reason": text(
                fo_decision.get("fo_reason"),
                "FO decision indicates no-trade environment"
            ),
            "risk_state": "blocked",
            "python_authoritative": {
                "no_trade": True,
                "strategy_a": strat_a,
                "strategy_b": strat_b
            },
            "llm_analysis": {
                "trend": trend_data,
                "fo": fo_data,
                "synthesis": synthesis_data
            },
            "merge_trace": {
                "used_python_override": True,
                "used_fo_risk_override": False,
                "used_fallback": status != "ok",
                "override_reason": "fo_no_trade",
                "llm_roles_available": {
                    "trend": trend_available,
                    "fo": fo_available,
                    "synthesis": synthesis_available
                }
            }
        }

    risk_state = "normal"
    final_action = ActionType.WAIT
    final_action_label = "Wait"
    final_reason = "Strategy levels available"

    # Priority 2: derive action from Python strategy A first, then B
    final_action, final_action_label = infer_action_from_strategy(strat_a, trend_data)
    if final_action == ActionType.WAIT:
        final_action, final_action_label = infer_action_from_strategy(strat_b, trend_data)

    # Optional synthesis direction hint, only if Python-derived action is still WAIT
    if final_action == ActionType.WAIT and synthesis_available:
        action_plan = as_dict(synthesis_data.get("action_plan"))
        direction = text(action_plan.get("direction"), "").lower()
        if direction == "long":
            final_action = ActionType.LONG
            final_action_label = ActionType.LONG
        elif direction == "short":
            final_action = ActionType.SHORT
            final_action_label = ActionType.SHORT

    # Priority 3: FO risk override
    used_fo_risk_override = False
    fo_sentiment = text(fo_data.get("sentiment"), "Neutral")

    if fo_sentiment == "Bearish" and final_action == ActionType.LONG:
        risk_state = "cautious"
        final_action = ActionType.WAIT
        final_action_label = "Wait for confirmation"
        used_fo_risk_override = True
    elif fo_sentiment == "Bullish" and final_action == ActionType.SHORT:
        risk_state = "cautious"
        final_action = ActionType.WAIT
        final_action_label = "Wait for confirmation"
        used_fo_risk_override = True

    # Priority 4: final reason
    final_reason = text(
        synthesis_data.get("final_verdict"),
        text(
            fo_decision.get("fo_reason"),
            "Analysis complete" if status == "ok" else "Deterministic strategy levels available"
        )
    )

    return {
        "status": status,
        "final_action": final_action,
        "final_action_label": final_action_label,
        "final_reason": final_reason,
        "risk_state": risk_state,
        "python_authoritative": {
            "no_trade": False,
            "strategy_a": strat_a,
            "strategy_b": strat_b
        },
        "llm_analysis": {
            "trend": trend_data,
            "fo": fo_data,
            "synthesis": synthesis_data
        },
        "merge_trace": {
            "used_python_override": False,
            "used_fo_risk_override": used_fo_risk_override,
            "used_fallback": status != "ok",
            "action_source": (
                "python_strategy"
                if final_action_label in [ActionType.LONG, ActionType.SHORT, "Wait"]
                else "llm_synthesis"
            ),
            "llm_roles_available": {
                "trend": trend_available,
                "fo": fo_available,
                "synthesis": synthesis_available
            }
        }
    }

def get_trend_prompt(context):
    return f"""
You are a senior trend analyst for a professional trading desk. Based ONLY on the numeric data below, provide a comprehensive trend analysis.

## INPUT DATA
{context}

## YOUR TASK
Analyze the trend structure, multi-timeframe alignment, Bollinger Band extremes, and key levels. Return ONLY valid JSON.

## ANALYSIS RULES

### 1. Bias Determination (D_Regime)
- "Bullish" → Uptrend confirmed
- "Bearish" → Downtrend confirmed  
- "Range" → Sideways market
- "RetailChop" → "choppy sideways market" (translate in output, but keep original in regime field)

### 2. Market Stage Mapping
| D_Regime | ADX | Market Stage |
|----------|-----|--------------|
| Bullish | > 25 | Advancing |
| Bullish | < 25 | Accumulation |
| Bearish | > 25 | Declining |
| Bearish | < 25 | Distribution |
| Range/SmartRange | any | Accumulation |
| RetailChop | any | Distribution |

### 3. Trend Strength (ADX)
- ADX > 25 → "Strong"
- ADX 20-25 → "Moderate"  
- ADX < 20 → "Weak"

### 4. BOLLINGER BAND (3 STANDARD DEVIATIONS) - CRITICAL
**These rules MUST be followed exactly:**

| Position | Description | Trading Implication |
|----------|-------------|---------------------|
| Close ABOVE Upper BB | "Price is trading above the upper Bollinger Band (3 standard deviations)" | OVERBOUGHT/STRETCHED/EXHAUSTED → Take profits, avoid new aggressive longs, consider mean-reversion |
| Close BELOW Lower BB | "Price is trading below the lower Bollinger Band (3 standard deviations)" | OVERSOLD/CAPITULATION → Look for mean-reversion, avoid new aggressive shorts |
| Close INSIDE Bands | "Price is trading inside the Bollinger Bands" | Normal conditions |

**SPECIAL RULES:**
- If close is ABOVE upper BB → MUST say "price is currently trading above the upper band" (NOT "holds below")
- If close is ABOVE upper BB → Call it "stretched", "overbought", or "exhausted"
- If close is BELOW lower BB → Call it "washed out", "oversold", or "capitulation"
- If close is ABOVE upper BB in an uptrend → "Extreme move, watch for failure to sustain above"
- If close is BELOW lower BB in a downtrend → "Extreme move, watch for reversal back inside"

### 5. ATR (Average True Range) - VOLATILITY CONTEXT
- Use ATR to describe expected daily range
- "ATR is X, so expect Y points of daily movement"
- Wider ATR = larger stops needed, more slippage risk
- Narrower ATR = tighter stops possible, range-bound conditions

| ATR Percentile (vs historical) | Description |
|-------------------------------|-------------|
| > 80th percentile | "Expanding volatility", "wide ranges expected" |
| 20th-80th percentile | "Normal volatility" |
| < 20th percentile | "Contracting volatility", "tight ranges expected" |

### 6. Multi-Timeframe Alignment
- **Strong alignment**: Trend, Setup, Entry all same direction
- **Moderate alignment**: Trend matches one of Setup/Entry
- **Weak alignment**: Trend conflicts with Setup/Entry → "Pullback only" or "Wait"

### 7. EMA Stack Interpretation
- "bullish_stack" → Price above all EMAs, short > long → Strong uptrend
- "bearish_stack" → Price below all EMAs, short < long → Strong downtrend
- "mixed_stack" → Conflicting signals → Trend weakening

### 8. Darvas Box Rules
- **State "inside"** → Range-bound, trade edges
- **State "above_upper"** → Breakout confirmed, favor continuation
- **State "below_lower"** → Breakdown confirmed, caution on longs
- **Strength score**: <4 weak, 4-7 moderate, >7 strong

### 9. RSI Zones (for forecast context)
- >80 → "Overbought / stretched / overheated" (extreme)
- 70-80 → "Overbought" (warning)
- <20 → "Oversold / washed out" (extreme)
- 20-30 → "Oversold" (warning)
- 40-60 → "Mid-range / neutral"
- 30-40 or 60-70 → "Transition / leaning weak or strong"

### 10. Key Levels Logic
- **Support**: Nearest price level BELOW current price (swing low, Darvas lower, PDL, lower BB)
- **Resistance**: Nearest price level ABOVE current price (swing high, Darvas upper, PDH, upper BB)
- If multiple levels exist, take the CLOSEST to current price

## OUTPUT JSON STRUCTURE
{{
    "bias": "Bullish/Bearish/Range",
    "bias_description": "Clear English explanation (e.g., 'uptrend', 'downtrend', 'choppy sideways')",
    "market_stage": "Advancing/Declining/Accumulation/Distribution",
    "trend_strength": "Strong/Moderate/Weak",
    "bollinger_bands": {{
        "position": "above_upper/inside/below_lower",
        "description": "Price is trading above/below/inside the Bollinger Bands",
        "implication": "overbought/oversold/normal",
        "action": "take_profits/avoid_aggressive/normal"
    }},
    "atr": {{
        "value": number,
        "description": "expanding/normal/contracting",
        "daily_range_expectation": "Expect X points of daily movement"
    }},
    "multi_timeframe": {{
        "alignment": "Strong/Moderate/Weak",
        "trend_regime": "string",
        "setup_regime": "string", 
        "entry_regime": "string",
        "action": "trend_following/pullback_only/range_trading/wait"
    }},
    "key_levels": {{
        "support": [number],
        "resistance": [number],
        "nearest_support": number,
        "nearest_resistance": number,
        "bollinger_support": number,
        "bollinger_resistance": number
    }},
    "indicators": {{
        "rsi": number,
        "rsi_zone": "overbought/oversold/neutral/extreme_overbought/extreme_oversold",
        "adx": number,
        "ema_stack": "bullish_stack/bearish_stack/mixed_stack",
        "atr": number
    }},
    "darvas": {{
        "state": "inside/above_upper/below_lower",
        "strength": number,
        "reliability": "Low/Moderate/High"
    }},
    "forecast": "IF [condition at key level] THEN [expected move] ELSE [alternative]",
    "risk_note": "Brief warning about Bollinger extreme, ATR width, or RSI level"
}}

## FORECAST FORMAT EXAMPLES
- "IF price holds above 22851.70 support, THEN expect bounce to 23272.40 resistance"
- "IF price breaks below 22804.55, THEN expect continuation to 22515.25"
- "IF price is above upper Bollinger Band (26285.58), THEN expect reversion back inside"
- "IF price stays inside Darvas box (351.40-434.95), THEN trade range edges"

## RISK NOTE EXAMPLES
- "Bollinger Band extreme: price is stretched above upper band, risk of sharp reversal"
- "ATR is expanding (X), expect wider daily ranges, adjust position sizing"
- "RSI at 82 indicates overbought conditions, avoid chasing longs"

Use the exact values from the data. Do not invent numbers.
"""

def get_fo_prompt(context):
    return f"""
You are an F&O specialist for a professional trading desk.

Use ONLY the numeric data provided below.
Do NOT invent numbers.
Do NOT add markdown.
Do NOT add explanations outside JSON.
Return ONLY one valid JSON object.


INPUT DATA
{context}


TASK
Analyze:
1. Options sentiment
2. IV regime and skew
3. PCR and positioning
4. Greeks interpretation
5. Futures OI behavior
6. Trading risk and options advice
7. FO decision override


RULES

1. Sentiment
- Sentiment here means near-term options-market tone, not contrarian investing view.
- If PCR > 1.2 OR Put IV > Call IV + 5 => "Bearish"
- If PCR < 0.8 OR Call IV > Put IV + 5 => "Bullish"
- If 0.8 <= PCR <= 1.2 AND abs(Call IV - Put IV) < 5 => "Neutral"

2. IV Regime
- Use provided iv_regime if present.
- Validate against ATM IV:
  - IV < 15 => "Low"
  - 15 to 25 => "Normal"
  - > 25 => "High"
- If provided regime and derived regime differ, mention it in risk_assessment.

3. Positioning
- PCR > 1.5 => "Put-heavy"
- PCR 1.2 to 1.5 => "Put-heavy"
- PCR 0.8 to 1.2 => "Balanced"
- PCR 0.6 to 0.8 => "Call-heavy"
- PCR < 0.6 => "Call-heavy"

4. Skew
- put_skew => puts richer than calls, fear/hedging dominant
- call_skew => calls richer than puts, optimism/speculation dominant
- neutral => no strong skew

5. Gamma
- high/extreme => fast delta change, higher execution risk
- low/moderate => more stable delta behavior

6. Liquidity
- poor/fair => slippage risk, avoid size
- good/excellent => execution acceptable

7. Term Structure
- frontelevated => near-term uncertainty elevated
- normalcontango => normal upward term curve
- flat => little term premium

8. Futures OI
- long_buildup => bullish conviction increasing
- short_buildup => bearish conviction increasing
- short_covering => bearish positions closing, bullish signal
- long_unwinding => bullish positions closing, bearish signal

9. FO Decision Override
- If no_trade = true in input decision context, then:
  - sentiment must be "Neutral"
  - options_advice must be "No fresh positions recommended"
  - fo_decision.bias must be "no_trade"
  - fo_decision.conviction must be "LOW"
  - fo_decision.no_trade must be true

10. Output style
- Keep every explanation short and practical.
- Keep greeks_summary fields under 20 words each.
- Use exact numbers from input where relevant.
- If a value is unavailable, use a reasonable text fallback like "unknown" for strings, but do not invent numeric values.


RETURN THIS EXACT JSON SHAPE
{{
  "sentiment": "Bullish/Bearish/Neutral",
  "sentiment_reason": "short string",
  "iv_regime": "Low/Normal/High",
  "iv_analysis": {{
    "call": number,
    "put": number,
    "alignment": "aligned/skewed",
    "interpretation": "short string"
  }},
  "positioning": "Call-heavy/Put-heavy/Balanced",
  "pcr_analysis": {{
    "value": number,
    "trend": "build_up/unwinding/stable",
    "interpretation": "short string"
  }},
  "iv_skew": {{
    "type": "put_skew/call_skew/neutral",
    "strength": "mild/strong/extreme"
  }},
  "gamma_exposure": "low/moderate/high/extreme",
  "liquidity": "poor/fair/good/excellent",
  "volume_momentum": "rising/flat/falling",
  "term_structure": "frontelevated/normalcontango/flat",
  "greeks_summary": {{
    "delta_view": "short string explaining directional bias from ATM call/put delta",
    "gamma_view": "short string explaining whether delta can change quickly near ATM",
    "vega_view": "short string explaining sensitivity to implied volatility changes",
    "theta_view": "short string explaining time-decay pressure on long options"
  }},
  "futures": {{
    "oi_state": "long_buildup/short_buildup/short_covering/long_unwinding",
    "conviction": "upgrade/downgrade/neutral"
  }},
  "risk_assessment": "1 short sentence",
  "options_advice": "1 short sentence",
  "fo_decision": {{
    "bias": "bullish/bearish/neutral/no_trade",
    "conviction": "LOW/MEDIUM/HIGH",
    "no_trade": true
  }}
}}
"""

def get_synthesis_prompt(trend_json, fo_json, strategy_context, persona_key):
    return f"""You are a senior trading strategist. Create a concise trading game plan.

## INPUTS
Trend: {trend_json}
FO: {fo_json}
Strategy Levels: {strategy_context}
Persona: {persona_key}

## RULES
- Strategy levels are AUTHORITATIVE - use exact entry/stop/targets
- If no_trade=true → direction must be "Wait"
- For Swing persona: Weekly → Daily → 4H timeframe explainer
- Keep narrative under 400 words total

## OUTPUT FORMAT (VALID JSON ONLY - no markdown, no extra text)
{{
    "summary": "2 sentences: current market stage and key level to watch",
    "section_1_bias": "3 sentences: higher timeframe bias, market stage, IF-THEN forecast",
    "section_2_multitf": "3 sentences: Trend→Setup→Entry alignment with 2 specific levels",
    "section_3_stance": "3 sentences: what trades to take, what to avoid, with exact levels",
    "section_4_action_plan": "3 bullet points: Trend watch, Setup watch, Entry trigger",
    "section_5_psychology": "2 sentences: who is trapped and what options positioning shows",
    "section_6_watch": "2 sentences: 2 key levels to monitor and invalidation condition",
    "action_plan": {{
        "direction": "Long/Short/Wait",
        "direction_reason": "1 sentence",
        "entry_level": null,
        "stop_level": null,
        "targets": [],
        "risk_override": "normal"
    }},
    "final_verdict": "1 sentence: action to take and invalidation level"
}}

Use exact numbers from Strategy Levels. If a number isn't available, use null.
Do NOT add markdown formatting. Return ONLY the JSON object."""

def format_ui_output_production(merged_result):
    """Format merged production result for UI display with null-safe fallbacks."""

    def as_dict(value):
        return value if isinstance(value, dict) else {}

    def as_list(value):
        return value if isinstance(value, list) else []

    def safe_text(value, default="N/A"):
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            return value if value else default
        return str(value)

    def safe_num(value, default="N/A", decimals=2):
        if value is None:
            return default
        try:
            n = float(value)
            return f"{n:.{decimals}f}"
        except (TypeError, ValueError):
            return safe_text(value, default)

    def extract_targets(strategy):
        strategy = as_dict(strategy)
        targets = as_list(strategy.get("targets"))

        t1 = None
        t2 = None

        if len(targets) >= 1:
            t1 = targets[0]
        elif strategy.get("target1") is not None:
            t1 = strategy.get("target1")

        if len(targets) >= 2:
            t2 = targets[1]
        elif strategy.get("target2") is not None:
            t2 = strategy.get("target2")

        return safe_num(t1), safe_num(t2)

    def format_strategy_block(title, strategy, default_type):
        strategy = as_dict(strategy)
        t1, t2 = extract_targets(strategy)

        return f"""
## {title} ({safe_text(strategy.get('type'), default_type)})
- **Entry:** {safe_num(strategy.get('entry'))} | **Stop:** {safe_num(strategy.get('stop'))}
- **Targets:** {t1} → {t2}
- **Conviction:** {safe_text(strategy.get('conviction'))}
"""

    merged_result = as_dict(merged_result)

    llm_analysis = as_dict(merged_result.get("llm_analysis"))
    trend = as_dict(llm_analysis.get("trend"))
    fo = as_dict(llm_analysis.get("fo"))
    synthesis = as_dict(llm_analysis.get("synthesis"))

    python_authoritative = as_dict(merged_result.get("python_authoritative"))
    strat_a = as_dict(python_authoritative.get("strategy_a"))
    strat_b = as_dict(python_authoritative.get("strategy_b"))

    action_plan = as_dict(synthesis.get("action_plan"))

    status = safe_text(merged_result.get("status"), "fallback").lower()
    if status == "ok":
        status_badge = "✅"
        status_line = "LLM + Python analysis available."
    elif status == "partial":
        status_badge = "⚠️"
        status_line = "Partial LLM analysis available. Python strategy remains authoritative."
    else:
        status_badge = "🔧"
        status_line = "Fallback mode. Deterministic Python output is being shown."

    summary = safe_text(
        synthesis.get("summary"),
        "LLM narrative unavailable. See deterministic strategy levels below."
    )

    final_verdict = safe_text(
        synthesis.get("final_verdict"),
        merged_result.get("final_reason") or "Trade only with defined risk and strict execution."
    )

    output = f"""
{status_badge} **Analysis Status:** {status.upper()}
{status_line}

## Big Picture
{summary}

## Market Trend
- **Bias:** {safe_text(trend.get('bias'))} | **Stage:** {safe_text(trend.get('market_stage'))}
- **Strength:** {safe_text(trend.get('trend_strength'))} | **Regime:** {safe_text(trend.get('regime'))}
- **RSI:** {safe_num(trend.get('rsi'), decimals=2)} | **ADX:** {safe_num(trend.get('adx'), decimals=2)}
- **Forecast:** {safe_text(trend.get('forecast'))}

## Options & F&O Context
- **Sentiment:** {safe_text(fo.get('sentiment'))} | **IV Regime:** {safe_text(fo.get('iv_regime'))}
- **Positioning:** {safe_text(fo.get('positioning'))}
- **Risk:** {safe_text(fo.get('risk_assessment'))}
- **Options Advice:** {safe_text(fo.get('options_advice'))}

## Action Plan
- **Direction:** {safe_text(action_plan.get('direction'), merged_result.get('final_action_label') or 'Wait')}
- **Approach:** {safe_text(action_plan.get('entry_approach'), 'See strategies below')}
- **Risk Note:** {safe_text(action_plan.get('risk_note'), merged_result.get('final_reason') or 'Use strict stop losses')}

{format_strategy_block("Strategy A", strat_a, "Pullback")}

{format_strategy_block("Strategy B", strat_b, "Range-Edge Fade")}

## Final Verdict
{final_verdict}

---
**Risk State:** {safe_text(merged_result.get('risk_state'), 'normal')} | **Action:** {safe_text(merged_result.get('final_action_label'), 'Wait')}
""".strip()

    return output

def generate_trainer_explanation_production(precomputed, regimes, fo_context, strategies, current_price, persona_key):
    """
    ENHANCED single-prompt trainer explanation with ALL analytics.
    Used as fallback when analyze() output is insufficient.
    """
    
    import time
    import copy
    
    # ========== CONFIGURATION ==========
    GENAI_API_KEY = os.environ.get("GENAI_API_KEY") or ""
    GEMINI_FLASH_MODEL = "gemini-2.0-flash"
    NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or ""
    
    # ========== BUILD RICH VERIFIED ANCHOR ==========
    verified_lines = []
    verified_lines.append("=" * 80)
    verified_lines.append("VERIFIED DATA FROM ANALYSIS - DO NOT FABRICATE OR CHANGE ANYTHING")
    verified_lines.append("=" * 80)
    verified_lines.append("")
    
    # 1. Regimes
    trend_regime = regimes.get("Trend_Regime", "Range")
    setup_regime = regimes.get("Setup_Regime", "Range")
    entry_regime = regimes.get("Entry_Regime", "Range")
    
    verified_lines.append("MARKET REGIMES (DO NOT INVENT OR CHANGE):")
    verified_lines.append(f"  • Trend Regime: {trend_regime}")
    verified_lines.append(f"  • Setup Regime: {setup_regime}")
    verified_lines.append(f"  • Entry Regime: {entry_regime}")
    verified_lines.append("")
    
    # 2. Market Stage
    market_stage = regimes.get("Market_Stage", "Unknown")
    if market_stage == "Unknown":
        if trend_regime == "Bullish":
            market_stage = "Advancing"
        elif trend_regime == "Bearish":
            market_stage = "Declining"
        else:
            market_stage = "Accumulation"
    
    verified_lines.append(f"MARKET STAGE (TREND TF): {market_stage}")
    verified_lines.append("")
    
    # 3. Current Price
    verified_lines.append(f"CURRENT PRICE: {current_price}")
    verified_lines.append("")
    
    # 4. Strategy Levels (AUTHORITATIVE)
    strat_a = strategies.get("A", {})
    strat_b = strategies.get("B", {})
    
    verified_lines.append("TRADING STRATEGIES (EXACT LEVELS - DO NOT CHANGE):")
    verified_lines.append(f"  • Strategy A: Entry={strat_a.get('entry', 'N/A')} | Stop={strat_a.get('stop', 'N/A')} | Targets={strat_a.get('target1', 'N/A')} → {strat_a.get('target2', 'N/A')}")
    verified_lines.append(f"  • Strategy B: Entry={strat_b.get('entry', 'N/A')} | Stop={strat_b.get('stop', 'N/A')} | Targets={strat_b.get('target1', 'N/A')} → {strat_b.get('target2', 'N/A')}")
    verified_lines.append("")
    
    # 5. Daily Indicators
    daily_ind = precomputed.get("DAILY", {}).get("indicators", {})
    if daily_ind:
        verified_lines.append("DAILY INDICATORS (USE THESE EXACT VALUES):")
        verified_lines.append(f"  • RSI14: {daily_ind.get('D_RSI14', 'N/A')}")
        verified_lines.append(f"  • ADX14: {daily_ind.get('D_ADX14', 'N/A')}")
        verified_lines.append(f"  • ATR14: {daily_ind.get('D_ATR14', 'N/A')}")
        verified_lines.append(f"  • EMA Stack: {daily_ind.get('D_EMA_stack', 'mixed')}")
        
        # Bollinger Bands with status
        bb_hi = daily_ind.get('D_BB_hi')
        bb_lo = daily_ind.get('D_BB_lo')
        close = daily_ind.get('D_Close')
        if bb_hi and bb_lo and close:
            verified_lines.append(f"  • Bollinger Upper: {bb_hi:.2f}")
            verified_lines.append(f"  • Bollinger Lower: {bb_lo:.2f}")
            if close > bb_hi:
                verified_lines.append(f"  • Bollinger Status: Price ABOVE upper band → STRETCHED/OVERBOUGHT")
            elif close < bb_lo:
                verified_lines.append(f"  • Bollinger Status: Price BELOW lower band → WASHED OUT/OVERSOLD")
            else:
                pct = (close - bb_lo) / (bb_hi - bb_lo) * 100
                verified_lines.append(f"  • Bollinger Status: Inside bands ({pct:.0f}% from lower to upper)")
        verified_lines.append("")
    
    # 6. EMA Stack Comments
    weekly_ind = precomputed.get("WEEKLY", {}).get("indicators", {})
    daily_ema_comment = daily_ind.get("D_EMA_comment", "")
    weekly_ema_comment = weekly_ind.get("W_EMA_comment", "")
    if daily_ema_comment or weekly_ema_comment:
        verified_lines.append("EMA STACK ANALYSIS:")
        if daily_ema_comment:
            verified_lines.append(f"  • Daily: {daily_ema_comment}")
        if weekly_ema_comment:
            verified_lines.append(f"  • Weekly: {weekly_ema_comment}")
        verified_lines.append("")
    
    # 7. Relative Strength (Mansfield)
    rs_snapshot = {
        "daily_bucket": daily_ind.get("D_RS_bucket", "Neutral"),
        "daily_mansfield": daily_ind.get("D_RS_Mansfield"),
        "weekly_bucket": weekly_ind.get("W_RS_bucket", "Neutral"),
        "weekly_mansfield": weekly_ind.get("W_RS_Mansfield"),
    }
    if rs_snapshot.get("daily_bucket") or rs_snapshot.get("weekly_bucket"):
        verified_lines.append("RELATIVE STRENGTH (vs PRIMARY INDEX):")
        if rs_snapshot.get("daily_mansfield") is not None:
            verified_lines.append(f"  • Daily: {rs_snapshot['daily_bucket']} (Mansfield: {rs_snapshot['daily_mansfield']:+.3f})")
        else:
            verified_lines.append(f"  • Daily: {rs_snapshot['daily_bucket']}")
        if rs_snapshot.get("weekly_mansfield") is not None:
            verified_lines.append(f"  • Weekly: {rs_snapshot['weekly_bucket']} (Mansfield: {rs_snapshot['weekly_mansfield']:+.3f})")
        else:
            verified_lines.append(f"  • Weekly: {rs_snapshot['weekly_bucket']}")
        verified_lines.append("")
    
    # 8. RSI Divergence (Multi-Timeframe)
    divergences = []
    for tf in ["5M", "15M", "30M", "1H", "4H", "DAILY", "WEEKLY"]:
        tf_data = precomputed.get(tf, {})
        if tf_data:
            ms_block = tf_data.get("market_structure", {})
            div_type = ms_block.get("rsi_divergence_type")
            div_strength = ms_block.get("rsi_divergence_strength")
            if div_type and div_type != "none":
                divergences.append(f"  • {tf}: {div_type.upper()} (strength: {div_strength:.2f})")
    
    if divergences:
        verified_lines.append("RSI DIVERGENCE (MULTI-TIMEFRAME):")
        verified_lines.extend(divergences)
        verified_lines.append("")
    
    # 9. Darvas Box Context (with strength)
    for tf in ["DAILY", "WEEKLY", "MONTHLY"]:
        tf_data = precomputed.get(tf, {})
        ms_block = tf_data.get("market_structure", {})
        darvas = ms_block.get("darvas_box", {})
        if darvas and darvas.get("is_valid_classical_darvas"):
            darvas_strength = ms_block.get("darvas_strength", {})
            verified_lines.append(f"DARVAS BOX ({tf} TF):")
            verified_lines.append(f"  • Upper: {darvas.get('upper', 'N/A'):.2f}")
            verified_lines.append(f"  • Lower: {darvas.get('lower', 'N/A'):.2f}")
            verified_lines.append(f"  • Mid: {darvas.get('mid', 'N/A'):.2f}")
            verified_lines.append(f"  • State: {darvas.get('state', 'inside')}")
            verified_lines.append(f"  • Strength: {darvas_strength.get('darvas_strength', 0):.1f}/10")
            verified_lines.append(f"  • Breakout Reliability: {darvas_strength.get('breakout_reliability', 'Low')}")
            verified_lines.append("")
            break
    
    # 10. Fibonacci Levels
    fib_data = {}
    for tf in ["DAILY", "WEEKLY", "MONTHLY"]:
        tf_data = precomputed.get(tf, {})
        ms_block = tf_data.get("market_structure", {})
        fib = ms_block.get("fib_levels", {})
        if fib:
            fib_data = fib
            break
    
    if fib_data:
        verified_lines.append("FIBONACCI LEVELS (DO NOT INVENT OR ADJUST):")
        for level, price in fib_data.items():
            verified_lines.append(f"  • {level}%: {price:.2f}")
        verified_lines.append("")
    
    # 11. GANN Metrics (if available)
    gann_metrics = precomputed.get("GANN_METRICS", {})
    if gann_metrics:
        verified_lines.append("=" * 80)
        verified_lines.append("GANN METRICS (SUPPORTING CONTEXT ONLY):")
        verified_lines.append("=" * 80)
        
        weekly = gann_metrics.get("weekly_patterns", {})
        if weekly.get("friday_weekly_high"):
            verified_lines.append(f"  • Friday made weekly high → Next week bias {weekly.get('next_week_bias', 'Bullish')} ({weekly.get('confidence', 65)}% confidence)")
        if weekly.get("friday_weekly_low"):
            verified_lines.append(f"  • Friday made weekly low → Next week bias {weekly.get('next_week_bias', 'Bearish')} ({weekly.get('confidence', 65)}% confidence)")
        
        monthly = gann_metrics.get("monthly_patterns", {})
        if monthly.get("double_top"):
            verified_lines.append(f"  • Monthly DOUBLE TOP detected → {monthly.get('signal', 'BEARISH')} signal")
        if monthly.get("triple_top"):
            verified_lines.append(f"  • Monthly TRIPLE TOP detected → {monthly.get('signal', 'BEARISH')} signal")
        if monthly.get("double_bottom"):
            verified_lines.append(f"  • Monthly DOUBLE BOTTOM detected → {monthly.get('signal', 'BULLISH')} signal")
        if monthly.get("triple_bottom"):
            verified_lines.append(f"  • Monthly TRIPLE BOTTOM detected → {monthly.get('signal', 'BULLISH')} signal")
        
        qtr = gann_metrics.get("quarterly_breakout", {})
        if qtr.get("breakout_above"):
            verified_lines.append(f"  • Quarterly breakout above {qtr.get('previous_quarter_high')} → BULLISH reversal")
        if qtr.get("breakdown_below"):
            verified_lines.append(f"  • Quarterly breakdown below {qtr.get('previous_quarter_low')} → BEARISH reversal")
        
        breakout = gann_metrics.get("breakout_patterns", {})
        if breakout.get("three_day_high_signal"):
            verified_lines.append(f"  • 3-day high broken → Expect 4th day surge")
        
        ma_break = gann_metrics.get("ma_break", {})
        if ma_break.get("ma_break_signal"):
            verified_lines.append(f"  • {ma_break.get('consecutive_days_below')} consecutive days below 30 DMA → Correction expected")
        
        verified_lines.append("")
    
    # 12. FO Metrics for F&O Persona
    if persona_key in ("fno", "fo"):
        fo_root = precomputed.get("FO_METRICS", {})
        front = fo_root.get("front", {})
        fo_signals = front.get("fo_signals", {})
        
        fo_lines = []
        fo_lines.append("=" * 80)
        fo_lines.append("OPTIONS & FUTURES METRICS (READ-ONLY – DO NOT INVENT):")
        fo_lines.append("=" * 80)
        
        if front.get("atm_iv_call"):
            fo_lines.append(f"  • ATM Call IV: {front.get('atm_iv_call'):.2f}")
        if front.get("atm_iv_put"):
            fo_lines.append(f"  • ATM Put IV: {front.get('atm_iv_put'):.2f}")
        if front.get("pcr_oi"):
            fo_lines.append(f"  • PCR OI: {front.get('pcr_oi'):.2f}")
        if fo_signals.get("delta_bias"):
            fo_lines.append(f"  • Delta Bias: {fo_signals.get('delta_bias')}")
        if fo_signals.get("gamma_exposure"):
            fo_lines.append(f"  • Gamma Exposure: {fo_signals.get('gamma_exposure')}")
        if fo_signals.get("skew_type"):
            fo_lines.append(f"  • Skew: {fo_signals.get('skew_type')}")
        if fo_signals.get("liquidity_grade"):
            fo_lines.append(f"  • Liquidity: {fo_signals.get('liquidity_grade')}")
        
        fo_lines.append("")
        verified_lines.extend(fo_lines)
    
    # 13. Determine allowed direction (HARD RULE)
    if trend_regime == "Bullish":
        allowed_direction = "LONG only"
        bias_text = "bullish"
        opposite_warning = "Do NOT recommend Short positions"
    elif trend_regime == "Bearish":
        allowed_direction = "SHORT only"
        bias_text = "bearish"
        opposite_warning = "Do NOT recommend Long positions"
    else:
        allowed_direction = "LONG or SHORT (range edges)"
        bias_text = "range-bound"
        opposite_warning = "Trade only at range edges"
    
    verified_anchor = "\n".join(verified_lines)
    
    # ========== TIMEFRAME EXPLAINER ==========
    tf_map = {
        "intraday": "Daily → 30m → 5m",
        "swing": "Weekly → Daily → 4H",
        "positional": "Monthly → Weekly → Daily",
        "fno": "Daily → 30m → 5m",
        "fo": "Daily → 30m → 5m",
        "investing": "Quarterly → Monthly → Weekly"
    }
    tf_explainer = tf_map.get(persona_key, "Weekly → Daily → 4H")
    
    persona_label_map = {
        "intraday": "Intraday", "swing": "Swing", "positional": "Positional",
        "fno": "F&O", "fo": "F&O", "investing": "Investing"
    }
    persona_label = persona_label_map.get(persona_key, persona_key)
    
    # ========== BUILD SINGLE COMPREHENSIVE PROMPT ==========
    prompt = f"""You are a professional trading strategist. Write a clear, actionable trading plan.

{verified_anchor}

## CRITICAL RULES (MUST FOLLOW - NO EXCEPTIONS):

1. **DIRECTION RULE (HARD CONSTRAINT):**
   - Trend Regime is {trend_regime}
   - Allowed direction: {allowed_direction}
   - {opposite_warning}
   
2. **MARKET STAGE RULE:**
   - Market Stage is {market_stage}
   - Advancing → Only long trades
   - Declining → Only short trades
   - Accumulation → Long bias
   - Distribution → Short bias

3. **STRATEGY LEVELS ARE AUTHORITATIVE:**
   - Strategy A: Entry={strat_a.get('entry', 'N/A')}, Stop={strat_a.get('stop', 'N/A')}, Targets={strat_a.get('target1', 'N/A')} → {strat_a.get('target2', 'N/A')}
   - Strategy B: Entry={strat_b.get('entry', 'N/A')}, Stop={strat_b.get('stop', 'N/A')}, Targets={strat_b.get('target1', 'N/A')} → {strat_b.get('target2', 'N/A')}
   - Use THESE EXACT numbers. NEVER invent new prices.

4. **GANN SIGNALS (if present):**
   - Use ONLY as confirmation
   - Do NOT let GANN override the primary regime

5. **RSI DIVERGENCE:**
   - Use as supporting context only
   - Do NOT override primary regime

## YOUR TASK:

Write a trading game plan for a {persona_label} trader using {tf_explainer} timeframe logic.

**Required Sections:**

1) **Big Picture** (2-3 sentences)
   - State: "{bias_text} bias with {market_stage} stage"
   - Mention any GANN confirmation if present
   - Give IF-THEN forecast using Strategy levels

2) **Key Levels** (list format)
   - Entry: {strat_a.get('entry', 'N/A')} or {strat_b.get('entry', 'N/A')}
   - Stop: {strat_a.get('stop', 'N/A')} or {strat_b.get('stop', 'N/A')}
   - Targets: {strat_a.get('target1', 'N/A')} → {strat_a.get('target2', 'N/A')}

3) **Action Plan** (bullet points)
   - Direction: {allowed_direction}
   - Entry trigger: Price at {strat_a.get('entry', 'N/A')} or {strat_b.get('entry', 'N/A')}
   - Stop placement: {strat_a.get('stop', 'N/A')} or {strat_b.get('stop', 'N/A')}

4) **What to Avoid**
   - Don't trade against {bias_text} bias
   - Don't chase breakouts without confirmation

5) **Risk Note**
   - ATR is {daily_ind.get('D_ATR14', 'N/A')}
   - Keep stops at least 1.5x ATR

6) **Bottom Line**
   - One sentence: {allowed_direction} using Strategy levels above

Be direct, use IF-THEN language, and use EXACT numbers from Strategy levels. No markdown. No jargon. Write for a {persona_label} trader.
"""
    
    # ========== SINGLE LLM CALL ==========
    explanation = None
    
    # Try Gemini
    if GENAI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GENAI_API_KEY)
            model = genai.GenerativeModel(GEMINI_FLASH_MODEL)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 1500,
                }
            )
            if response and response.text:
                explanation = response.text
                print("✅ Gemini single-prompt succeeded")
        except Exception as e:
            print(f"Gemini failed: {e}")
    
    # Try NVIDIA as fallback
    if not explanation and NVIDIA_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=NVIDIA_API_KEY,
                timeout=120.0
            )
            response = client.chat.completions.create(
                model="meta/llama-3.2-3b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            explanation = response.choices[0].message.content
            print("✅ NVIDIA single-prompt succeeded")
        except Exception as e:
            print(f"NVIDIA failed: {e}")
    
    # Final fallback (always works)
    if not explanation:
        explanation = f"""
**Big Picture**
{bias_text.capitalize()} bias with {market_stage} stage. Use Strategy levels below.

**Key Levels**
- Entry: {strat_a.get('entry', 'N/A')} or {strat_b.get('entry', 'N/A')}
- Stop: {strat_a.get('stop', 'N/A')} or {strat_b.get('stop', 'N/A')}
- Targets: {strat_a.get('target1', 'N/A')} → {strat_a.get('target2', 'N/A')}

**Action Plan**
Direction: {allowed_direction}
Entry at {strat_a.get('entry', 'N/A')} with stop at {strat_a.get('stop', 'N/A')}

**Risk Note**
ATR is {daily_ind.get('D_ATR14', 'N/A')}. Keep stops appropriate.

**Bottom Line**
{allowed_direction} using Strategy levels above.
"""
        print("⚠️ Using fallback template")
    
    return explanation

# ======================================================
# Flask Routes
# ======================================================
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_symbol():
    """Run analysis for a symbol"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        mode = data.get('mode', 'Intraday')
        
        if not symbol:
            return jsonify({'error': 'Please enter a valid symbol'}), 400
        
        user_symbol = symbol
        if user_symbol in SYMBOL_ALIASES:
            user_symbol = SYMBOL_ALIASES[user_symbol]
        
        mode_lower = mode.strip().lower()
        if mode_lower == "f&o":
            persona_key = "fno"
        else:
            persona_key = mode_lower
        
        is_index = (user_symbol in INDEX_SYMBOLS) or (user_symbol in NO_PARENT_INDEX_ETFS)
        
        logger.info(f"Analyzing {user_symbol} with mode {mode} (persona: {persona_key})")
        
        # ========== STEP 1: Compute precomputed data ==========
        precomputed = compute_precomputed(user_symbol, persona=persona_key)
                # DEBUG: Check what's in precomputed
        print("\n" + "=" * 80)
        print("DEBUG: precomputed keys available:")
        print("=" * 80)
        print(f"Keys: {list(precomputed.keys())}")
        
        # Check specific sections
        if "DAILY" in precomputed:
            daily = precomputed["DAILY"]
            print(f"\nDAILY keys: {list(daily.keys())}")
            if "indicators" in daily:
                print(f"Daily indicators: {list(daily['indicators'].keys())[:10]}...")
        
        if "GANN_METRICS" in precomputed:
            print(f"\nGANN_METRICS keys: {list(precomputed['GANN_METRICS'].keys())}")
        
        if "FO_METRICS" in precomputed:
            print(f"\nFO_METRICS keys: {list(precomputed['FO_METRICS'].keys())}")
        
        print("=" * 80 + "\n")

        # ========== STEP 1.5: Extract ATR and volatility metrics ==========
        atr_metrics = {
            "daily_atr": None,
            "daily_atr_percent": None,
            "weekly_atr": None,
            "weekly_atr_percent": None,
            "volatility_regime": "normal",
            "atr_display": "--",           # ✅ ADD THIS
            "atr_percent_display": "--"    # ✅ ADD THIS
        }

        try:
            daily_block = precomputed.get("DAILY", {})
            daily_ind = daily_block.get("indicators", {})
            
            # Get Daily ATR
            daily_atr = daily_ind.get("D_ATR14")
            if daily_atr is not None:
                daily_atr_float = float(daily_atr)
                atr_metrics["daily_atr"] = round(daily_atr_float, 2)
                
                # Calculate ATR as percentage of price
                current_price = daily_ind.get("D_Close") or daily_ind.get("Close")
                if current_price and current_price > 0:
                    atr_percent = (daily_atr_float / float(current_price)) * 100
                    atr_metrics["daily_atr_percent"] = round(atr_percent, 2)
                    atr_metrics["atr_percent_display"] = f"{round(atr_percent, 2)}%"
                    
                    # Determine volatility regime
                    if atr_percent < 1:
                        atr_metrics["volatility_regime"] = "low"
                    elif atr_percent < 2:
                        atr_metrics["volatility_regime"] = "normal"
                    else:
                        atr_metrics["volatility_regime"] = "high"
                
                # ✅ Format display string (CRITICAL for frontend)
                atr_metrics["atr_display"] = f"{round(daily_atr_float, 2)}"
                if atr_metrics.get("daily_atr_percent"):
                    atr_metrics["atr_display"] += f" ({atr_metrics['daily_atr_percent']}%)"
            
            # Get Weekly ATR (if available)
            weekly_block = precomputed.get("WEEKLY", {})
            weekly_ind = weekly_block.get("indicators", {})
            weekly_atr = weekly_ind.get("W_ATR14")
            if weekly_atr is not None:
                atr_metrics["weekly_atr"] = round(float(weekly_atr), 2)
                
                weekly_price = weekly_ind.get("W_Close")
                if weekly_price and weekly_price > 0:
                    weekly_atr_percent = (float(weekly_atr) / float(weekly_price)) * 100
                    atr_metrics["weekly_atr_percent"] = round(weekly_atr_percent, 2)
                    
        except Exception as e:
            print(f"DEBUG: ATR extraction error: {e}")
            atr_metrics = {
                "daily_atr": None,
                "daily_atr_percent": None,
                "weekly_atr": None,
                "weekly_atr_percent": None,
                "volatility_regime": "normal",
                "atr_display": "--",
                "atr_percent_display": "--"
            }

        print(f"DEBUG: ATR Metrics = {atr_metrics}")
        
        # ========== STEP 2: Run analysis ==========
        output_text, regimes, darvas_state_ui = analyze(
            user_symbol=user_symbol,
            mode=persona_key,
            precomputed=precomputed
        )

                # DEBUG: Print first 2000 chars of output_text
        print("\n" + "=" * 80)
        print("DEBUG: output_text (first 2000 chars):")
        print("=" * 80)
        print(output_text[:2000])
        print("=" * 80 + "\n")
        
        # Also check if STRATEGIES_JSON exists
        if "STRATEGIES_JSON" in output_text:
            print("DEBUG: STRATEGIES_JSON found in output_text")
            # Find and print the JSON part
            idx = output_text.find("STRATEGIES_JSON")
            print(output_text[idx:idx+500])
        else:
            print("DEBUG: STRATEGIES_JSON NOT found in output_text")
        
        if not isinstance(output_text, str):
            return jsonify({'error': 'Unexpected response from trading engine'}), 500
        
        # ========== STEP 3: Extract current price ==========
        current_price = None
        try:
            daily_block = precomputed.get("DAILY", {})
            daily_ind = daily_block.get("indicators", {})
            current_price = daily_ind.get("D_Close") or daily_ind.get("Close")
            print(f"DEBUG: Current price = {current_price}")
        except Exception as e:
            print(f"DEBUG: Could not extract current_price: {e}")
            current_price = None
        
        # ========== ADD MARKET STAGE ==========
        # Get Market_Stage from analysis
        market_stage = regimes.get("Market_Stage")

        # Ensure market_stage always has a value
        if market_stage is None or market_stage == "":
            print(f"DEBUG WARNING: No Market_Stage provided by analysis. Available regimes keys: {list(regimes.keys())}")
            market_stage = "Unknown"
        else:
            print(f"DEBUG: Market Stage from analysis: {market_stage}")

        # Now market_stage is guaranteed to have a value (either from analysis or "Unknown")
        
        # ========== STEP 4: Extract final strategies ==========
        # Try JSON format first, then fallback to text format
        final_strategies = extract_final_strategies_from_output(output_text)
        if not final_strategies:
            final_strategies = extract_strategies_from_text(output_text)
            if final_strategies:
                print("DEBUG: Extracted strategies from text format")
            else:
                print("DEBUG: No strategies found in either JSON or text format")
        
                # DEBUG: Print what we got
        print("\n" + "=" * 80)
        print("DEBUG: final_strategies content:")
        print("=" * 80)
        print(final_strategies)
        print("=" * 80 + "\n")
        
        # ========== INITIALIZE strategy_a AND strategy_b ==========
        strategy_a = {}
        strategy_b = {}
        if final_strategies:
            strategy_a = final_strategies.get("A", {})
            strategy_b = final_strategies.get("B", {})
        
        # ========== STEP 4.5: EXTRACT MISSING DATA FROM RAW OUTPUT ==========
        raw_data_anchor = ""
        
        # Extract Darvas Box from raw output
        darvas_direct = extract_darvas_direct(output_text)
        darvas_anchor = ""
        if darvas_direct:
            print("DEBUG: Direct Darvas extraction successful:", darvas_direct)
            darvas_anchor = "=" * 80 + "\n"
            darvas_anchor += "DARVAS BOX CONTEXT (INSTITUTIONAL STRUCTURE - DO NOT INVENT):\n"
            darvas_anchor += "=" * 80 + "\n"
            for key, value in darvas_direct.items():
                darvas_anchor += f"  • {key}: {value}\n"
            darvas_anchor += "\n"
        
        # Extract RSI Divergence
        rsi_anchor = extract_rsi_divergence_from_raw(output_text)
        if rsi_anchor:
            print("DEBUG: RSI Divergence extracted successfully")
        
        # Extract Fibonacci levels
        fib_anchor = extract_fibonacci_from_raw(output_text)
        if fib_anchor:
            print("DEBUG: Fibonacci levels extracted successfully")
        
        # Extract EMA stack
        ema_anchor = extract_ema_stack_from_raw(output_text)
        if ema_anchor:
            print("DEBUG: EMA stack extracted successfully")
        
        # Combine all anchors (without GANN for now)
        raw_data_anchor = darvas_anchor + rsi_anchor + fib_anchor + ema_anchor
        if raw_data_anchor:
            print(f"DEBUG: Raw data anchor length: {len(raw_data_anchor)}")

        # ========== STEP 5: Clean output ==========
        clean_output = output_text
        idx_json = clean_output.rfind("STRATEGIES_JSON")
        if idx_json != -1:
            clean_output = clean_output[:idx_json].rstrip()
        
        if "LLM output missing or invalid STRATEGIES_JSON" in clean_output:
            clean_output = clean_output[:clean_output.rfind("LLM output missing or invalid STRATEGIES_JSON")].rstrip()
        
        clean_output = fmt_text_prices(clean_output)
        
        # Simplify internal labels
        idx_strategies = clean_output.rfind("📌 Strategy Suggestions")
        if idx_strategies != -1:
            narrative_only = clean_output[:idx_strategies]
            strategies_section = clean_output[idx_strategies:]
            narrative_only = simplify_internal_labels(narrative_only)
            clean_output = narrative_only + strategies_section
        else:
            clean_output = simplify_internal_labels(clean_output)
        
        # ========== STEP 6: Extract verified prices anchor from cleaned output ==========
        # Add debug to see what we're working with
        print("=" * 80)
        print("DEBUG: Checking clean_output content:")
        print("=" * 80)
        print(f"Contains 'Darvas': {'Darvas' in clean_output}")
        print(f"Contains 'upper:': {'upper:' in clean_output}")
        print(f"Contains 'Strategy A Entry': {'Strategy A Entry' in clean_output}")
        print(f"Contains 'Strategy A:': {'Strategy A:' in clean_output}")
        print(f"Clean output length: {len(clean_output)}")
        print("=" * 80)

        # Try to extract from cleaned output (may be empty)
        verified_prices_anchor = extract_verified_prices(clean_output)
        # ✅ ADD MARKET STAGE TO VERIFIED ANCHOR (NO HARDCODING)
        # Get market_stage from regimes (already calculated)
        market_stage = regimes.get("Market_Stage")

        # Create market stage anchor section
        market_stage_anchor = ""
        if market_stage:
            market_stage_anchor = f"""
================================================================================
MARKET STAGE (WEINSTEIN STAGE) - FROM SYSTEM CALCULATION:
================================================================================
• MARKET_STAGE_TREND_TF: {market_stage}

CRITICAL: This is the EXACT market stage determined by the system's institutional logic.
You MUST use this value in your explanation. DO NOT change it.
- "Accumulation" = base building / consolidation phase
- "Advancing" = markup / uptrend phase
- "Distribution" = top building phase
- "Declining" = markdown / downtrend phase

If the analysis text below mentions a different stage, IGNORE it and use this value.
================================================================================

"""
        else:
            print("DEBUG WARNING: market_stage not available to add to verified anchor")

        # Prepend market stage anchor to verified_prices_anchor
        if market_stage_anchor:
            verified_prices_anchor = market_stage_anchor + (verified_prices_anchor or "")
        
        # ========== STEP 7: Build strategy override (MANDATORY PRICE LEVELS) ==========
        strategy_override = ""

        # First try to get from final_strategies (using strategy_a and strategy_b already defined)
        if strategy_a.get('entry', 'N/A') != 'N/A' or strategy_b.get('entry', 'N/A') != 'N/A':
            strategy_override = f"""
**===== MANDATORY PRICE LEVELS FROM ANALYSIS =====**

Strategy A ({strategy_a.get('type', 'Pullback')}):
  Entry: {strategy_a.get('entry', 'N/A')}
  Stop Loss: {strategy_a.get('stop', 'N/A')}
  Target 1: {strategy_a.get('target1', 'N/A')}
  Target 2: {strategy_a.get('target2', 'N/A')}
  Conviction: {strategy_a.get('conviction', 'N/A')}

Strategy B ({strategy_b.get('type', 'Range-Edge Fade')}):
  Entry: {strategy_b.get('entry', 'N/A')}
  Stop Loss: {strategy_b.get('stop', 'N/A')}
  Target 1: {strategy_b.get('target1', 'N/A')}
  Target 2: {strategy_b.get('target2', 'N/A')}
  Conviction: {strategy_b.get('conviction', 'N/A')}

Current Price: {current_price if current_price else 'N/A'}

YOU MUST USE THESE EXACT NUMBERS. DO NOT INVENT NEW PRICE LEVELS.
"""
            print("DEBUG: Strategy override built from final_strategies")
        else:
            print("DEBUG: No valid strategy entries in final_strategies")

        # If still empty, try to extract directly from output_text
        if not strategy_override and "Strategy A" in output_text:
            print("DEBUG: Attempting direct extraction from output_text")
            direct_strategies = extract_strategies_from_text(output_text)
            if direct_strategies:
                strategy_a = direct_strategies.get("A", {})
                strategy_b = direct_strategies.get("B", {})
                strategy_override = f"""
**===== MANDATORY PRICE LEVELS FROM ANALYSIS =====**

Strategy A ({strategy_a.get('type', 'Pullback')}):
  Entry: {strategy_a.get('entry', 'N/A')}
  Stop Loss: {strategy_a.get('stop', 'N/A')}
  Target 1: {strategy_a.get('target1', 'N/A')}
  Target 2: {strategy_a.get('target2', 'N/A')}
  Conviction: {strategy_a.get('conviction', 'N/A')}

Strategy B ({strategy_b.get('type', 'Range-Edge Fade')}):
  Entry: {strategy_b.get('entry', 'N/A')}
  Stop Loss: {strategy_b.get('stop', 'N/A')}
  Target 1: {strategy_b.get('target1', 'N/A')}
  Target 2: {strategy_b.get('target2', 'N/A')}
  Conviction: {strategy_b.get('conviction', 'N/A')}

Current Price: {current_price if current_price else 'N/A'}

YOU MUST USE THESE EXACT NUMBERS. DO NOT INVENT NEW PRICE LEVELS.
"""
                print("DEBUG: Strategy override built from direct extraction")
        
        # ========== STEP 8: COMBINE ALL ANCHORS (MOVED AFTER STEP 7) ==========
        # Priority: Raw data anchor (Darvas, RSI, Fibonacci) + Verified prices + Strategy override
        combined_anchor = raw_data_anchor + verified_prices_anchor + strategy_override

        # If still empty, create emergency anchor with at least current price and regimes
        if not combined_anchor or len(combined_anchor.strip()) < 100:
            print("DEBUG: WARNING - combined_anchor is empty or too short!")
            
            # Get ATR from precomputed
            atr_value = "N/A"
            try:
                daily_block = precomputed.get("DAILY", {})
                daily_ind = daily_block.get("indicators", {})
                atr_value = daily_ind.get("D_ATR14", "N/A")
                if atr_value and atr_value != "N/A":
                    atr_value = f"{float(atr_value):.2f}"
            except Exception:
                pass
            
            combined_anchor = f"""
**CRITICAL MARKET DATA (PLEASE USE THESE):**

Current Price: {current_price if current_price else 'N/A'}
ATR (Volatility): {atr_value}

Market Regimes:
- Trend: {regimes.get('Trend_Regime', 'N/A')}
- Setup: {regimes.get('Setup_Regime', 'N/A')}
- Entry: {regimes.get('Entry_Regime', 'N/A')}

Market Stage: {market_stage if market_stage else regimes.get('Market_Stage', 'Accumulation')}

Strategy Levels from Analysis:
Strategy A: Entry {strategy_a.get('entry', 'N/A')}, Stop {strategy_a.get('stop', 'N/A')}, Targets {strategy_a.get('target1', 'N/A')}, {strategy_a.get('target2', 'N/A')}
Strategy B: Entry {strategy_b.get('entry', 'N/A')}, Stop {strategy_b.get('stop', 'N/A')}, Targets {strategy_b.get('target1', 'N/A')}, {strategy_b.get('target2', 'N/A')}

YOU MUST USE THESE EXACT NUMBERS IN YOUR ANALYSIS.
"""        
        # ========== STEP 9: Print debug info ==========
        print("=" * 80)
        print("DEBUG: Verified anchor content:")
        print("=" * 80)
        if verified_prices_anchor:
            print(verified_prices_anchor[:2000])
        else:
            print("EMPTY")
        print("=" * 80)
        print(f"Verified anchor length: {len(verified_prices_anchor)}")
        print(f"Raw data anchor length: {len(raw_data_anchor)}")
        print(f"Strategy override length: {len(strategy_override)}")
        print(f"Combined anchor length: {len(combined_anchor)}")
        print("=" * 80)
        
        # ========== STEP 10: F&O Metrics Extraction ==========
        fo_context = {
            "iv_call": None,
            "iv_put": None,
            "iv_call_change": None,
            "iv_put_change": None,
            "pcr": None,
            "pcr_change": None,
            "call_delta": None,
            "put_delta": None,
            "call_gamma": None,
            "put_gamma": None,
            "call_vega": None,
            "put_vega": None,
            "call_theta": None,
            "put_theta": None,
            "total_call_oi": None,
            "total_put_oi": None,
            "call_oi_change": None,
            "put_oi_change": None,
            "oi_trend": None,
            "futures_state": None,
            "futures_price": None,
            "futures_price_change": None,
            "futures_oi": None,
            "futures_oi_change": None,
            "futures_basis": None,
            "term_structure": None,
            "delta_bias": None,
            "gamma_exposure": None,
            "skew_type": None,
            "skew_strength": None,
            "liquidity_grade": None,
            "volume_momentum": None,
            "fo_bias": None,
            "fo_conviction": None,
            "fo_option_style": None,
            "fo_risk_profile": None,
            "fo_no_trade": False,  # ✅ ADD THIS LINE
        }
        
        fo_snapshot = ""
        fo_decision_snapshot = ""
        
        if persona_key in ("fno", "fo") and isinstance(precomputed, dict):
            try:
                fo_root = precomputed.get("FO_METRICS") or {}
                front = fo_root.get("front") or {}
                fut_1h = fo_root.get("futures_1h") or {}
                fo_decision = precomputed.get("FO_DECISION") or {}
                
                # Extract front expiry metrics
                fo_context["iv_call"] = front.get("atm_iv_call")
                fo_context["iv_put"] = front.get("atm_iv_put")
                fo_context["iv_call_change"] = front.get("atm_iv_call_change")
                fo_context["iv_put_change"] = front.get("atm_iv_put_change")
                fo_context["pcr"] = front.get("pcr_oi")
                fo_context["pcr_change"] = front.get("pcr_oi_change")
                fo_context["call_delta"] = front.get("atm_ce_delta")
                fo_context["put_delta"] = front.get("atm_pe_delta")
                fo_context["call_gamma"] = front.get("atm_ce_gamma")
                fo_context["put_gamma"] = front.get("atm_pe_gamma")
                fo_context["call_vega"] = front.get("atm_ce_vega")
                fo_context["put_vega"] = front.get("atm_pe_vega")
                fo_context["call_theta"] = front.get("atm_ce_theta")
                fo_context["put_theta"] = front.get("atm_pe_theta")
                fo_context["total_call_oi"] = front.get("total_call_oi")
                fo_context["total_put_oi"] = front.get("total_put_oi")
                fo_context["call_oi_change"] = front.get("total_call_oi_change")
                fo_context["put_oi_change"] = front.get("total_put_oi_change")
                fo_context["oi_trend"] = front.get("oi_trend")
                fo_context["term_structure"] = fo_root.get("term_structure")
                
                # Extract FO signals
                fo_signals = front.get("fo_signals") or {}
                fo_context["delta_bias"] = fo_signals.get("delta_bias")
                fo_context["gamma_exposure"] = fo_signals.get("gamma_exposure")
                fo_context["skew_type"] = fo_signals.get("skew_type")
                fo_context["skew_strength"] = fo_signals.get("skew_strength")
                fo_context["liquidity_grade"] = fo_signals.get("liquidity_grade")
                fo_context["volume_momentum"] = fo_signals.get("volume_momentum")
                
                # Extract futures 1H metrics
                fo_context["futures_state"] = fut_1h.get("fut_1h_oi_state")
                fo_context["futures_price"] = fut_1h.get("fut_1h_price_now")
                fo_context["futures_price_change"] = fut_1h.get("fut_1h_price_change")
                fo_context["futures_oi"] = fut_1h.get("fut_1h_oi_now")
                fo_context["futures_oi_change"] = fut_1h.get("fut_1h_oi_change")
                fo_context["futures_basis"] = fut_1h.get("futures_basis")
                
                # Extract FO_DECISION
                if fo_decision:
                    fo_context["fo_bias"] = fo_decision.get("fo_bias")
                    fo_context["fo_conviction"] = fo_decision.get("fo_conviction")
                    fo_context["fo_option_style"] = fo_decision.get("fo_option_style")
                    fo_context["fo_risk_profile"] = fo_decision.get("fo_risk_profile")
                    fo_context["fo_no_trade"] = fo_decision.get("fo_no_trade", False)  # ✅ ADD THIS LINE
                
                # Build FO snapshot for trainer prompt
                lines = []
                lines.append("OPTIONS & FUTURES SNAPSHOT (DO NOT INVENT ANY NUMBERS):")
                
                # IV section
                iv_parts = []
                if fo_context["iv_call"] is not None:
                    iv_parts.append(f"ATM Call IV: {fo_context['iv_call']:.2f}")
                if fo_context["iv_put"] is not None:
                    iv_parts.append(f"ATM Put IV: {fo_context['iv_put']:.2f}")
                if iv_parts:
                    lines.append("  • " + " | ".join(iv_parts))
                
                # IV changes
                iv_chg_parts = []
                if fo_context["iv_call_change"] is not None:
                    iv_chg_parts.append(f"Call IV change: {fo_context['iv_call_change']:+.2f}")
                if fo_context["iv_put_change"] is not None:
                    iv_chg_parts.append(f"Put IV change: {fo_context['iv_put_change']:+.2f}")
                if iv_chg_parts:
                    lines.append("  • " + " | ".join(iv_chg_parts))
                
                # Term structure
                if fo_context["term_structure"]:
                    term_map = {
                        "normalcontango": "normal contango (later expiries have higher IV)",
                        "frontelevated": "front-month elevated (near-term uncertainty)",
                        "flat": "flat term structure"
                    }
                    term_desc = term_map.get(str(fo_context["term_structure"]).lower(), fo_context["term_structure"])
                    lines.append(f"  • Term structure: {term_desc}")
                
                # PCR
                if fo_context["pcr"] is not None:
                    lines.append(f"  • PCR OI: {fo_context['pcr']:.2f}")
                if fo_context["pcr_change"] is not None:
                    lines.append(f"  • PCR change (today): {fo_context['pcr_change']:+.2f}")
                
                # Greeks
                greek_parts = []
                if fo_context["call_delta"] is not None:
                    greek_parts.append(f"Call Δ: {fo_context['call_delta']:.2f}")
                if fo_context["put_delta"] is not None:
                    greek_parts.append(f"Put Δ: {fo_context['put_delta']:.2f}")
                if greek_parts:
                    lines.append("  • " + " | ".join(greek_parts))
                
                gamma_parts = []
                if fo_context["call_gamma"] is not None:
                    gamma_parts.append(f"Call Γ: {fo_context['call_gamma']:.4f}")
                if fo_context["put_gamma"] is not None:
                    gamma_parts.append(f"Put Γ: {fo_context['put_gamma']:.4f}")
                if gamma_parts:
                    lines.append("  • " + " | ".join(gamma_parts))
                
                vega_parts = []
                if fo_context["call_vega"] is not None:
                    vega_parts.append(f"Call ν: {fo_context['call_vega']:.2f}")
                if fo_context["put_vega"] is not None:
                    vega_parts.append(f"Put ν: {fo_context['put_vega']:.2f}")
                if vega_parts:
                    lines.append("  • " + " | ".join(vega_parts))
                
                theta_parts = []
                if fo_context["call_theta"] is not None:
                    theta_parts.append(f"Call Θ: {fo_context['call_theta']:.2f}")
                if fo_context["put_theta"] is not None:
                    theta_parts.append(f"Put Θ: {fo_context['put_theta']:.2f}")
                if theta_parts:
                    lines.append("  • " + " | ".join(theta_parts))
                
                # OI totals
                oi_parts = []
                if fo_context["total_call_oi"] is not None:
                    oi_parts.append(f"Call OI: {fo_context['total_call_oi']:,.0f}")
                if fo_context["total_put_oi"] is not None:
                    oi_parts.append(f"Put OI: {fo_context['total_put_oi']:,.0f}")
                if oi_parts:
                    lines.append("  • " + " | ".join(oi_parts))
                
                # OI changes
                oi_chg_parts = []
                if fo_context["call_oi_change"] is not None:
                    oi_chg_parts.append(f"Call OI change: {fo_context['call_oi_change']:+.0f}")
                if fo_context["put_oi_change"] is not None:
                    oi_chg_parts.append(f"Put OI change: {fo_context['put_oi_change']:+.0f}")
                if oi_chg_parts:
                    lines.append("  • " + " | ".join(oi_chg_parts))
                
                if fo_context["oi_trend"]:
                    lines.append(f"  • OI trend: {fo_context['oi_trend']}")
                
                # Delta bias
                if fo_context["delta_bias"]:
                    lines.append(f"  • Delta bias: {fo_context['delta_bias']}")
                
                # Gamma exposure
                if fo_context["gamma_exposure"]:
                    lines.append(f"  • Gamma exposure: {fo_context['gamma_exposure']}")
                
                # Skew
                if fo_context["skew_type"]:
                    skew_str = f"  • Skew: {fo_context['skew_type']}"
                    if fo_context["skew_strength"]:
                        skew_str += f" ({fo_context['skew_strength']})"
                    lines.append(skew_str)
                
                # Liquidity
                if fo_context["liquidity_grade"]:
                    lines.append(f"  • Liquidity: {fo_context['liquidity_grade']}")
                
                # Volume momentum
                if fo_context["volume_momentum"]:
                    lines.append(f"  • Volume momentum: {fo_context['volume_momentum']}")
                
                # Futures 1H
                fut_parts = []
                if fo_context["futures_state"]:
                    fut_parts.append(f"OI state: {fo_context['futures_state']}")
                if fo_context["futures_price"] is not None:
                    fut_parts.append(f"Price: {fo_context['futures_price']:.2f}")
                if fo_context["futures_price_change"] is not None:
                    fut_parts.append(f"Δ: {fo_context['futures_price_change']:+.2f}")
                if fut_parts:
                    lines.append("  • 1H Futures: " + " | ".join(fut_parts))
                
                if fo_context["futures_oi"] is not None or fo_context["futures_oi_change"] is not None:
                    fut_oi_parts = []
                    if fo_context["futures_oi"] is not None:
                        fut_oi_parts.append(f"OI: {fo_context['futures_oi']:,.0f}")
                    if fo_context["futures_oi_change"] is not None:
                        fut_oi_parts.append(f"OI change: {fo_context['futures_oi_change']:+.0f}")
                    if fut_oi_parts:
                        lines.append("  • " + " | ".join(fut_oi_parts))
                
                if fo_context["futures_basis"] is not None:
                    lines.append(f"  • Futures basis: {fo_context['futures_basis']:.2f}")
                
                if lines:
                    fo_snapshot = "\n".join(lines) + "\n"
                
                # FO_DECISION snapshot
                if fo_decision:
                    dec_lines = []
                    dec_lines.append("FO DECISION VIEW (DERIVED FROM FO_METRICS ONLY):")
                    
                    core_parts = []
                    if fo_context["fo_bias"]:
                        core_parts.append(f"Bias: {fo_context['fo_bias']}")
                    if fo_context["fo_conviction"]:
                        core_parts.append(f"Conviction: {fo_context['fo_conviction']}")
                    if fo_context["fo_option_style"]:
                        core_parts.append(f"Option style: {fo_context['fo_option_style']}")
                    if fo_context["fo_risk_profile"]:
                        core_parts.append(f"Risk profile: {fo_context['fo_risk_profile']}")
                    if core_parts:
                        dec_lines.append("  • " + " | ".join(core_parts))
                    
                    if len(dec_lines) > 1:
                        fo_decision_snapshot = "\n".join(dec_lines) + "\n"
                        
            except Exception as e:
                logger.debug(f"FO metrics extraction error: {e}")
        
        # ========== STEP 11: Build RS snapshot ==========
        rs_snapshot = {"daily_bucket": None, "weekly_bucket": None}
        if precomputed:
            daily_tf = precomputed.get("DAILY", {})
            if daily_tf:
                daily_ind = daily_tf.get("indicators", {})
                rs_snapshot["daily_bucket"] = daily_ind.get("D_RS_bucket")
                rs_snapshot["daily_mansfield"] = daily_ind.get("D_RS_Mansfield")
            weekly_tf = precomputed.get("WEEKLY", {})
            if weekly_tf:
                weekly_ind = weekly_tf.get("indicators", {})
                rs_snapshot["weekly_bucket"] = weekly_ind.get("W_RS_bucket")
                rs_snapshot["weekly_mansfield"] = weekly_ind.get("W_RS_Mansfield")
        
        # ========== STEP 12: Get GANN metrics ==========
        gann_metrics = precomputed.get("GANN_METRICS", {})

        # ========== DEBUG: Check gann_metrics before passing ==========
        print("=" * 80)
        print("DEBUG: gann_metrics content check:")
        print("=" * 80)
        if gann_metrics:
            print(f"gann_metrics keys: {list(gann_metrics.keys())}")
            # Check specific sections
            weekly = gann_metrics.get("weekly_patterns", {})
            print(f"weekly_patterns: {weekly}")
            monthly = gann_metrics.get("monthly_patterns", {})
            print(f"monthly_patterns: {monthly}")
            breakout = gann_metrics.get("breakout_patterns", {})
            print(f"breakout_patterns: {breakout}")
        else:
            print("❌ WARNING: gann_metrics is EMPTY or None!")
        print("=" * 80)

        # ========== STEP 12.5: Add GANN to combined anchor ==========
        gann_anchor = extract_gann_metrics_for_anchor(gann_metrics)
        if gann_anchor:
            print("DEBUG: GANN metrics added to anchor")
            combined_anchor = gann_anchor + combined_anchor
        
        # ========== STEP 13: Generate trainer explanation (convert technical → beginner-friendly) ==========
        
        # Call the Gemini trainer to convert technical analysis to beginner-friendly language
        try:
            trainer_explanation = generate_trainer_explanation(
                clean_output=clean_output,
                raw_output=output_text,  # ✅ Pass the raw engine output
                persona_key=persona_key,
                is_index=is_index,
                precomputed=precomputed,
                verified_prices_anchor=verified_prices_anchor,
                fo_snapshot=fo_snapshot,
                fo_decision_snapshot=fo_decision_snapshot,
                rs_snapshot=rs_snapshot,
                gann_metrics=gann_metrics,
                regimes=regimes,  # ✅ ADD THIS
                market_stage=market_stage  # ✅ Make sure this is passed correctly
            )
            print(f"DEBUG: Primary trainer succeeded - length: {len(trainer_explanation)} chars")
        except Exception as e:
            print(f"DEBUG: Primary trainer failed: {e}")
            trainer_explanation = None
        
        # If primary fails, use fallback (LLM-based)
        if not trainer_explanation or len(trainer_explanation) < 100:
            print("DEBUG: Primary trainer failed, using fallback")
            trainer_explanation = generate_trainer_explanation_production(
                precomputed, regimes, fo_context, final_strategies, current_price, persona_key
            )
            print(f"DEBUG: Fallback trainer length: {len(trainer_explanation)} chars")
        
        # Clean up the trainer explanation (remove strategy section, JSON, markdown)
        if "📌 Strategy Suggestions" in trainer_explanation:
            trainer_explanation = trainer_explanation.split("📌 Strategy Suggestions")[0].strip()
        if "STRATEGIES_JSON" in trainer_explanation:
            trainer_explanation = trainer_explanation.split("STRATEGIES_JSON")[0].strip()
        trainer_explanation = re.sub(r'```text\s*|\s*```', '', trainer_explanation)
        trainer_explanation = re.sub(r'```json\s*|\s*```', '', trainer_explanation)
        trainer_explanation = trainer_explanation.strip()

        # ========== STEP 14: Compute quick action ==========
        quick = compute_quick_action(mode, regimes, darvas_state_ui)
        
        # ========== STEP 15: Return response ==========
        return jsonify({
            'success': True,
            'explanation': trainer_explanation,  # Changed from 'explanation' to 'trainer_explanation'
            'quick_action': quick,
            'raw_analysis': clean_output,
            'timestamp': datetime.now(IST).strftime("%d-%b-%Y %I:%M%p"),
            'symbol': symbol,
            'mode': mode,
            'regimes': regimes,
            'fo_context': fo_context,
            'darvas_state': darvas_state_ui,
            'atr_metrics': atr_metrics  # ✅ ADD THIS LINE
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/export-pdf', methods=['POST'])
def export_pdf():
    """Export analysis as PDF"""
    try:
        from pdf_report import build_pdf
        
        data = request.get_json()
        symbol = data.get('symbol', '')
        mode = data.get('mode', '')
        explanation = data.get('explanation', '')
        quick_action = data.get('quick_action', {})
        strategies = data.get('strategies', {})
        
        clean_text = explanation.replace('**', '').replace('*', '')
        
        pdf_buffer = build_pdf(
            stock_name=symbol,
            raw_text=clean_text,
            strategies_json=strategies,
            quick_action=quick_action
        )
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"{symbol}_analysis.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"PDF export error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now(IST).isoformat()})

if __name__ == '__main__':
    app.run(debug=IS_DEV, host='0.0.0.0', port=5000)
