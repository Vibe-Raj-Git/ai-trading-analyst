# backend/gann_calculator.py
"""
GANN Rules Calculator for Stock/Index Analysis
Implements W.D. Gann's classic trading rules including:
- Friday weekly patterns (high/low)
- Tuesday Lows – Uptrend Marker
- Wednesday Highs – Downtrend Signal
- 4-week breakouts / 3-day high break
- Correction ratios (5:3, 9:5)
- Volume spikes in consolidation
- Monthly double/triple bottoms/tops
- Quarterly breakouts
- 30 DMA rule
- 100% Rise Resistance
- 50% Sell Zone

IMPORTANT: This module is designed for CONFIRMATION ONLY.
GANN signals should never override primary Smart Money regimes.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

# Optional: Set up logging (remove if not needed)
logger = logging.getLogger(__name__)


def calculate_gann_levels(high: float, low: float, close: float) -> Dict:
    """
    Calculate Gann's 8 price levels from a swing high/low.
    
    Gann's 1/8th, 2/8th, 3/8th, 4/8th, 5/8th, 6/8th, 7/8th, 8/8th levels
    """
    diff = high - low
    levels = {}
    
    for i in range(1, 9):
        level = low + (diff * i / 8)
        levels[f"{i}/8"] = round(level, 2)
    
    levels["Mid"] = round((high + low) / 2, 2)
    levels["Support"] = round(low, 2)
    levels["Resistance"] = round(high, 2)
    
    return levels


def calculate_gann_angles(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Calculate Gann's geometric angles (1x1, 2x1, 1x2, etc.)
    Returns empty dict if insufficient data or calculation fails.
    """
    if len(df) < lookback:
        return {}
    
    df_sub = df.tail(lookback).copy()
    angles = {}
    
    try:
        price_range = df_sub['high'].max() - df_sub['low'].min()
        time_range = (df_sub.index[-1] - df_sub.index[0]).days
        if time_range > 0 and price_range > 0:
            slope_1x1 = price_range / time_range
            
            angles = {
                "1x1_slope": round(slope_1x1, 4),
                "1x2_slope": round(slope_1x1 * 2, 4),
                "2x1_slope": round(slope_1x1 / 2, 4),
                "1x4_slope": round(slope_1x1 * 4, 4),
                "4x1_slope": round(slope_1x1 / 4, 4),
                "is_uptrend": df_sub['close'].iloc[-1] > df_sub['close'].iloc[0]
            }
    except Exception as e:
        # Log silently; return empty dict is safe
        logger.debug(f"GANN angles calculation failed: {e}")
        angles = {}
    
    return angles


def detect_gann_weekly_pattern(df_weekly: pd.DataFrame) -> Dict:
    """
    Detect Gann's Friday/Weekly patterns:
    1. Friday Weekly High → Bullish Next Week
    2. Friday Weekly Low → Bearish Next Week
    
    Note: Tuesday/Wednesday patterns (Tuesday lows in uptrend, 
          Wednesday highs in downtrend) are now implemented
          in detect_day_of_week_patterns().
    
    Returns dict with:
        - friday_weekly_high: bool
        - friday_weekly_low: bool
        - next_week_bias: "BULLISH" / "BEARISH" / "NEUTRAL"
        - confidence: int (0-100)
    """
    if df_weekly is None or df_weekly.empty or len(df_weekly) < 2:
        return {}
    
    last_week = df_weekly.iloc[-1]
    
    result = {
        "friday_weekly_high": False,
        "friday_weekly_low": False,
        "next_week_bias": "NEUTRAL",
        "confidence": 0
    }
    
    weekly_high = last_week['high']
    weekly_low = last_week['low']
    friday_close = last_week['close']
    
    # Use range-based proximity for "near high/low"
    weekly_range = weekly_high - weekly_low
    
    # Check if Friday close is near weekly high (top 5% of range)
    if weekly_range > 0:
        if friday_close >= weekly_high - (weekly_range * 0.05):
            result["friday_weekly_high"] = True
            result["next_week_bias"] = "BULLISH"
            result["confidence"] = 65
        
        # Check if Friday close is near weekly low (bottom 5% of range)
        elif friday_close <= weekly_low + (weekly_range * 0.05):
            result["friday_weekly_low"] = True
            result["next_week_bias"] = "BEARISH"
            result["confidence"] = 65
    
    return result


def detect_day_of_week_patterns(df_daily: pd.DataFrame, df_weekly: pd.DataFrame = None) -> Dict:
    """
    Detect Gann's day-of-week patterns:
    3. Tuesday Lows – Uptrend Marker: In a highly uptrending market, weekly low is achieved on Tuesday
    4. Wednesday Highs – Downtrend Signal: If market is in a strong downtrend, weekly highs are on Wednesday
    
    Uses daily data to identify which day of the week made the weekly extreme.
    
    Returns dict with:
        - tuesday_low_in_uptrend: bool
        - wednesday_high_in_downtrend: bool
        - week_of_pattern: str (date of the week)
        - confidence: int (0-100)
    """
    if df_daily is None or df_daily.empty or len(df_daily) < 10:
        return {}
    
    result = {
        "tuesday_low_in_uptrend": False,
        "wednesday_high_in_downtrend": False,
        "week_of_pattern": None,
        "confidence": 0
    }
    
    try:
        # Get the last 4 weeks of daily data
        last_date = df_daily.index[-1]
        start_date = last_date - pd.Timedelta(days=28)  # 4 weeks
        
        df_recent = df_daily[df_daily.index >= start_date].copy()
        
        # Add day of week column
        df_recent['day_of_week'] = df_recent.index.dayofweek  # Monday=0, Tuesday=1, Wednesday=2, ..., Friday=4
        
        # Group by week
        df_recent['week'] = df_recent.index.isocalendar().week
        df_recent['year'] = df_recent.index.isocalendar().year
        df_recent['week_key'] = df_recent['year'].astype(str) + '-' + df_recent['week'].astype(str)
        
        weeks_analyzed = 0
        tuesday_low_count = 0
        wednesday_high_count = 0
        
        for week_key, week_data in df_recent.groupby('week_key'):
            if len(week_data) < 3:
                continue
            
            # Find day of week for weekly low
            weekly_low_idx = week_data['low'].idxmin()
            weekly_low_day = week_data.loc[weekly_low_idx, 'day_of_week']
            
            # Find day of week for weekly high
            weekly_high_idx = week_data['high'].idxmax()
            weekly_high_day = week_data.loc[weekly_high_idx, 'day_of_week']
            
            # Check if this week was in an uptrend (price higher than previous week)
            prev_week_data = df_recent[df_recent['week_key'] < week_key].tail(5)
            is_uptrend_week = False
            is_downtrend_week = False
            
            if not prev_week_data.empty:
                prev_week_close = prev_week_data.iloc[-1]['close']
                current_week_close = week_data.iloc[-1]['close']
                is_uptrend_week = current_week_close > prev_week_close
                is_downtrend_week = current_week_close < prev_week_close
            
            # Tuesday low in uptrend
            if is_uptrend_week and weekly_low_day == 1:  # Tuesday = 1
                tuesday_low_count += 1
                weeks_analyzed += 1
                result["week_of_pattern"] = week_key
            
            # Wednesday high in downtrend
            if is_downtrend_week and weekly_high_day == 2:  # Wednesday = 2
                wednesday_high_count += 1
                weeks_analyzed += 1
                result["week_of_pattern"] = week_key
        
        if weeks_analyzed > 0:
            # Calculate confidence based on pattern frequency
            if tuesday_low_count >= 2:
                result["tuesday_low_in_uptrend"] = True
                result["confidence"] = min(85, 60 + (tuesday_low_count * 10))
            
            if wednesday_high_count >= 2:
                result["wednesday_high_in_downtrend"] = True
                result["confidence"] = min(85, 60 + (wednesday_high_count * 10))
                
    except Exception as e:
        logger.debug(f"Day of week patterns calculation failed: {e}")
    
    return result


def calculate_gann_fib_ratios(high: float, low: float) -> Dict:
    """
    Calculate Gann's key retracement ratios (not Fibonacci, but Gann's own)
    Gann used: 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 8/8
    Also used 1/3 and 2/3
    """
    diff = high - low
    
    return {
        "gann_1_8": round(low + diff * 0.125, 2),
        "gann_2_8": round(low + diff * 0.25, 2),
        "gann_3_8": round(low + diff * 0.375, 2),
        "gann_4_8": round(low + diff * 0.5, 2),
        "gann_5_8": round(low + diff * 0.625, 2),
        "gann_6_8": round(low + diff * 0.75, 2),
        "gann_7_8": round(low + diff * 0.875, 2),
        "gann_8_8": round(high, 2),
        "gann_1_3": round(low + diff / 3, 2),
        "gann_2_3": round(low + diff * 2 / 3, 2),
        "gann_50": round((high + low) / 2, 2),
    }


def detect_breakout_patterns(df: pd.DataFrame) -> Dict:
    """
    Detect Gann's breakout patterns:
    - 4-Week High Breakout (using 20 trading days)
    - 4-Week Low Breakdown
    - 3-Day High Break → Fourth-Day Surge
    
    Returns dict with:
        - four_week_high_break: price or None
        - four_week_low_break: price or None
        - three_day_high_signal: bool
        - next_day_surge_expected: bool
        - stop_gann: price or None
    """
    if df.empty or len(df) < 20:
        return {}
    
    result = {
        "four_week_high_break": None,
        "four_week_low_break": None,
        "three_day_high_break": None,
        "three_day_high_signal": False,
        "next_day_surge_expected": False,
        "stop_gann": None
    }
    
    # 4-Week High/Low (20 trading days)
    if len(df) >= 20:
        last_20_high = df['high'].tail(20).max()
        last_20_low = df['low'].tail(20).min()
        current_close = df['close'].iloc[-1]
        
        if current_close > last_20_high:
            result["four_week_high_break"] = current_close
        elif current_close < last_20_low:
            result["four_week_low_break"] = current_close
    
    # 3-Day High Break → Fourth-Day Surge
    if len(df) >= 4:
        last_3_days_high = df['high'].tail(4).iloc[:-1].max()
        current_close = df['close'].iloc[-1]
        
        if current_close > last_3_days_high:
            result["three_day_high_break"] = current_close
            result["three_day_high_signal"] = True
            result["next_day_surge_expected"] = True
            # Gann's stop: 3 points below the 3-day high
            result["stop_gann"] = round(last_3_days_high - 3, 2)
    
    return result


def detect_correction_ratios(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect Gann's correction ratios:
    - 5:3 rise-correction
    - 9:5 rise-correction
    - Deeper correction warning (when subsequent corrections are larger)
    
    Returns dict with:
        - consecutive_up_days: int
        - consecutive_down_days: int
        - expected_correction_days: int
        - correction_ratio_detected: "5:3" / "9:5" / None
        - deeper_correction_warning: bool
    """
    if len(df) < lookback + 10:
        return {}
    
    result = {
        "consecutive_up_days": 0,
        "consecutive_down_days": 0,
        "expected_correction_days": 0,
        "correction_ratio_detected": None,
        "deeper_correction_warning": False
    }
    
    closes = df['close'].values
    up_days = 0
    down_days = 0
    last_trend = None
    
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            if last_trend == "up":
                up_days += 1
            else:
                up_days = 1
                down_days = 0
            last_trend = "up"
        elif closes[i] < closes[i-1]:
            if last_trend == "down":
                down_days += 1
            else:
                down_days = 1
                up_days = 0
            last_trend = "down"
    
    result["consecutive_up_days"] = up_days
    result["consecutive_down_days"] = down_days
    
    # Check 9:5 first (higher ratio), then 5:3
    if up_days >= 9:
        result["expected_correction_days"] = min(5, int(up_days * 0.56))
        result["correction_ratio_detected"] = "9:5"
    elif up_days >= 5:
        result["expected_correction_days"] = min(3, int(up_days * 0.6))
        result["correction_ratio_detected"] = "5:3"
    
    # Check for deeper correction warning
    corrections = []
    in_correction = False
    correction_start = None
    correction_end = None
    
    for i in range(1, len(closes)):
        if closes[i] < closes[i-1] and not in_correction:
            in_correction = True
            correction_start = i
        elif closes[i] > closes[i-1] and in_correction:
            in_correction = False
            correction_end = i
            correction_length = correction_end - correction_start
            if correction_start is not None and correction_start < len(closes):
                min_price = min(closes[correction_start:correction_end+1])
                if closes[correction_start] > 0:
                    correction_depth = (closes[correction_start] - min_price) / closes[correction_start]
                else:
                    correction_depth = 0
                corrections.append({"length": correction_length, "depth": correction_depth})
    
    if len(corrections) >= 2:
        last = corrections[-1]
        previous = corrections[-2]
        if last["length"] > previous["length"] and last["depth"] > previous["depth"]:
            result["deeper_correction_warning"] = True
    
    return result


def detect_volume_signals(df: pd.DataFrame) -> Dict:
    """
    Gann's volume spike rule:
    In choppy/consolidation phase, volume spike signals trend change
    
    Returns dict with:
        - volume_spike_detected: bool
        - consolidation_phase: bool (uses ADX < 20 if available)
        - spike_magnitude: float (current_volume / avg_volume)
        - signal_strength: "LOW" / "MEDIUM" / "HIGH"
    """
    if df.empty or len(df) < 20:
        return {}
    
    result = {
        "volume_spike_detected": False,
        "consolidation_phase": False,
        "spike_magnitude": 0,
        "signal_strength": "LOW"
    }
    
    # Check if in consolidation (low ADX, if column exists)
    if 'ADX' in df.columns:
        adx = df['ADX'].iloc[-1]
        result["consolidation_phase"] = adx < 20 if not pd.isna(adx) else False
    
    # Check volume spike
    avg_volume = df['volume'].tail(20).mean()
    current_volume = df['volume'].iloc[-1]
    
    if avg_volume > 0 and current_volume > avg_volume * 1.5:
        result["volume_spike_detected"] = True
        result["spike_magnitude"] = round(current_volume / avg_volume, 2)
        
        if result["spike_magnitude"] > 2.5:
            result["signal_strength"] = "HIGH"
        elif result["spike_magnitude"] > 2.0:
            result["signal_strength"] = "MEDIUM"
    
    return result


def detect_monthly_patterns(df_monthly: pd.DataFrame) -> Dict:
    """
    Detect Gann's monthly patterns:
    - Double/Triple bottom/top with 6+ month gap
    
    Returns dict with:
        - double_bottom: bool
        - double_top: bool
        - triple_bottom: bool
        - triple_top: bool
        - gap_months: int
        - signal: "BULLISH" / "BEARISH" / "STRONG_BULLISH" / "STRONG_BEARISH" / "NEUTRAL"
    """
    if df_monthly is None or df_monthly.empty or len(df_monthly) < 12:
        return {}
    
    result = {
        "double_bottom": False,
        "double_top": False,
        "triple_bottom": False,
        "triple_top": False,
        "gap_months": 0,
        "signal": "NEUTRAL"
    }
    
    closes = df_monthly['close'].values
    
    # Detect swing lows
    swing_lows = []
    for i in range(1, len(closes)-1):
        if closes[i] < closes[i-1] and closes[i] < closes[i+1]:
            swing_lows.append({"index": i, "price": closes[i], "date": df_monthly.index[i]})
    
    # Detect swing highs
    swing_highs = []
    for i in range(1, len(closes)-1):
        if closes[i] > closes[i-1] and closes[i] > closes[i+1]:
            swing_highs.append({"index": i, "price": closes[i], "date": df_monthly.index[i]})
    
    # Check for double/triple bottoms
    if len(swing_lows) >= 2:
        last_low = swing_lows[-1]
        for prev_low in swing_lows[-3:-1]:
            month_gap = abs((last_low["date"] - prev_low["date"]).days) / 30
            price_diff = abs(last_low["price"] - prev_low["price"]) / last_low["price"] if last_low["price"] > 0 else 1
            
            if month_gap >= 6 and price_diff < 0.03:
                result["double_bottom"] = True
                result["gap_months"] = int(month_gap)
                result["signal"] = "BULLISH"
                
                # Check for triple bottom
                if len(swing_lows) >= 3:
                    third_low = swing_lows[-3]
                    month_gap_2 = abs((last_low["date"] - third_low["date"]).days) / 30
                    if month_gap_2 >= 6:
                        result["triple_bottom"] = True
                        result["signal"] = "STRONG_BULLISH"
    
    # Check for double/triple tops
    if len(swing_highs) >= 2:
        last_high = swing_highs[-1]
        for prev_high in swing_highs[-3:-1]:
            month_gap = abs((last_high["date"] - prev_high["date"]).days) / 30
            price_diff = abs(last_high["price"] - prev_high["price"]) / last_high["price"] if last_high["price"] > 0 else 1
            
            if month_gap >= 6 and price_diff < 0.03:
                result["double_top"] = True
                result["gap_months"] = int(month_gap)
                result["signal"] = "BEARISH"
                
                if len(swing_highs) >= 3:
                    third_high = swing_highs[-3]
                    month_gap_2 = abs((last_high["date"] - third_high["date"]).days) / 30
                    if month_gap_2 >= 6:
                        result["triple_top"] = True
                        result["signal"] = "STRONG_BEARISH"
    
    return result


def detect_quarterly_breakout(df_quarterly: pd.DataFrame) -> Dict:
    """
    Gann's quarterly breakout rule:
    Crossing previous quarter's high/low signals trend reversal
    
    Returns dict with:
        - previous_quarter_high: price or None
        - previous_quarter_low: price or None
        - breakout_above: bool
        - breakdown_below: bool
        - trend_reversal_signal: bool
        - direction: "BULLISH" / "BEARISH" / None
    """
    if df_quarterly is None or df_quarterly.empty or len(df_quarterly) < 2:
        return {}
    
    result = {
        "previous_quarter_high": None,
        "previous_quarter_low": None,
        "breakout_above": False,
        "breakdown_below": False,
        "trend_reversal_signal": False,
        "direction": None
    }
    
    current = df_quarterly.iloc[-1]
    previous = df_quarterly.iloc[-2]
    
    result["previous_quarter_high"] = round(previous['high'], 2)
    result["previous_quarter_low"] = round(previous['low'], 2)
    
    if current['close'] > previous['high']:
        result["breakout_above"] = True
        result["trend_reversal_signal"] = True
        result["direction"] = "BULLISH"
    elif current['close'] < previous['low']:
        result["breakdown_below"] = True
        result["trend_reversal_signal"] = True
        result["direction"] = "BEARISH"
    
    return result


def detect_gann_ma_break(df_daily: pd.DataFrame, ma_period: int = 30) -> Dict:
    """
    Gann's 30 DMA rule:
    In uptrend, 2 consecutive days below 30 DMA signals major correction
    
    Note: The "previous_trend_up" heuristic uses close[-10] < close[-5]
          as a simple proxy. This is a heuristic, not a strict Gann rule.
    
    Returns dict with:
        - ma_break_signal: bool
        - consecutive_days_below: int
        - consecutive_days_above: int
        - correction_expected: bool
        - trend_reversal_warning: bool
    """
    if df_daily is None or df_daily.empty or len(df_daily) < ma_period:
        return {}
    
    result = {
        "ma_break_signal": False,
        "consecutive_days_below": 0,
        "consecutive_days_above": 0,
        "correction_expected": False,
        "trend_reversal_warning": False
    }
    
    ma = df_daily['close'].rolling(ma_period).mean()
    
    # Check last 5 days
    last_5 = df_daily.tail(5)
    days_below = 0
    days_above = 0
    
    for idx, row in last_5.iterrows():
        if not pd.isna(ma.loc[idx]):
            if row['close'] < ma.loc[idx]:
                days_below += 1
            elif row['close'] > ma.loc[idx]:
                days_above += 1
    
    result["consecutive_days_below"] = days_below
    result["consecutive_days_above"] = days_above
    
    # Simple heuristic: was the trend up in the last 10 days?
    # This checks if price 10 days ago was lower than 5 days ago
    if len(df_daily) >= 10:
        previous_trend_up = df_daily['close'].iloc[-10] < df_daily['close'].iloc[-5] if not df_daily.empty else False
        
        if days_below >= 2 and previous_trend_up:
            result["ma_break_signal"] = True
            result["correction_expected"] = True
            
            if days_below >= 3:
                result["trend_reversal_warning"] = True
    
    return result


def detect_100_percent_resistance(df_daily: pd.DataFrame) -> Dict:
    """
    Gann's 100% Rise Resistance Rule:
    When price starts rising from a particular level, Rs.100 or 100% rise
    whichever is earlier becomes a strong resistance.
    
    Returns dict with:
        - key_level: price of the swing low
        - one_hundred_percent_level: 2x the swing low
        - distance_to_100pct: distance to 100% level as percentage
        - is_near_resistance: bool (within 2% of 100% level)
        - is_above_100pct: bool (price above 100% level)
    """
    if df_daily.empty or len(df_daily) < 20:
        return {}
    
    result = {
        "key_level": None,
        "one_hundred_percent_level": None,
        "distance_to_100pct": None,
        "is_near_resistance": False,
        "is_above_100pct": False
    }
    
    try:
        # Find significant swing low in recent history (last 100 days)
        recent_lows = df_daily['low'].tail(100)
        key_low = recent_lows.min()
        
        # Calculate 100% level
        hundred_pct_level = key_low * 2
        
        # Check current price relative to 100% level
        current_price = df_daily['close'].iloc[-1]
        distance = abs(current_price - hundred_pct_level) / hundred_pct_level
        
        result["key_level"] = round(key_low, 2)
        result["one_hundred_percent_level"] = round(hundred_pct_level, 2)
        result["distance_to_100pct"] = round(distance * 100, 2)
        result["is_near_resistance"] = distance < 0.02  # Within 2%
        result["is_above_100pct"] = current_price > hundred_pct_level
        
    except Exception as e:
        logger.debug(f"100% resistance calculation failed: {e}")
    
    return result


def detect_50_percent_sell_zone(df_daily: pd.DataFrame) -> Dict:
    """
    Gann's 50% Sell Zone Rule:
    50% of the last highest selling price is the strong support area.
    Any stock trading below this 50% level is not that useful for investment.
    
    Returns dict with:
        - last_high: the highest price in recent history
        - fifty_percent_level: 50% of last high
        - is_below_50_percent: bool
        - investment_suitable: bool (true if above 50% level)
        - distance_from_50pct: percentage distance
    """
    if df_daily.empty or len(df_daily) < 50:
        return {}
    
    result = {
        "last_high": None,
        "fifty_percent_level": None,
        "is_below_50_percent": False,
        "investment_suitable": True,
        "distance_from_50pct": None
    }
    
    try:
        # Find the highest price in the last 50 days
        last_high = df_daily['high'].tail(50).max()
        fifty_percent = last_high / 2
        
        current_price = df_daily['close'].iloc[-1]
        
        result["last_high"] = round(last_high, 2)
        result["fifty_percent_level"] = round(fifty_percent, 2)
        result["is_below_50_percent"] = current_price < fifty_percent
        result["investment_suitable"] = current_price >= fifty_percent
        result["distance_from_50pct"] = round(((current_price - fifty_percent) / fifty_percent) * 100, 2)
        
    except Exception as e:
        logger.debug(f"50% sell zone calculation failed: {e}")
    
    return result


def calculate_all_gann_metrics(
    df_daily: pd.DataFrame,
    df_weekly: pd.DataFrame,
    df_monthly: pd.DataFrame,
    df_quarterly: pd.DataFrame,
    current_price: float
) -> Dict:
    """
    Calculate all GANN metrics for integration into precomputed dict
    
    Args:
        df_daily: Daily OHLCV DataFrame
        df_weekly: Weekly OHLCV DataFrame
        df_monthly: Monthly OHLCV DataFrame
        df_quarterly: Quarterly OHLCV DataFrame
        current_price: Current price for reference
    
    Returns:
        Dictionary with all GANN metrics. Empty dicts for missing calculations.
    """
    gann_metrics = {}
    
    # 1. GANN Price Levels from last swing
    if len(df_daily) > 20:
        swing_high = df_daily['high'].tail(20).max()
        swing_low = df_daily['low'].tail(20).min()
        if swing_high > swing_low:
            gann_metrics["price_levels"] = calculate_gann_levels(swing_high, swing_low, current_price)
            gann_metrics["fib_ratios"] = calculate_gann_fib_ratios(swing_high, swing_low)
    
    # 2. GANN Angles (silent on failure, returns empty dict)
    gann_metrics["angles"] = calculate_gann_angles(df_daily, lookback=50)
    
    # 3. Weekly Patterns (Friday only)
    if df_weekly is not None and not df_weekly.empty:
        gann_metrics["weekly_patterns"] = detect_gann_weekly_pattern(df_weekly)
    
    # 4. Day of Week Patterns (Tuesday Lows, Wednesday Highs)
    if df_daily is not None and not df_daily.empty:
        gann_metrics["day_of_week_patterns"] = detect_day_of_week_patterns(df_daily, df_weekly)
    
    # 5. Breakout Patterns
    gann_metrics["breakout_patterns"] = detect_breakout_patterns(df_daily)
    
    # 6. Correction Ratios
    gann_metrics["correction_ratios"] = detect_correction_ratios(df_daily)
    
    # 7. Volume Signals
    gann_metrics["volume_signals"] = detect_volume_signals(df_daily)
    
    # 8. Monthly Patterns
    if df_monthly is not None and not df_monthly.empty:
        gann_metrics["monthly_patterns"] = detect_monthly_patterns(df_monthly)
    
    # 9. Quarterly Breakout
    if df_quarterly is not None and not df_quarterly.empty:
        gann_metrics["quarterly_breakout"] = detect_quarterly_breakout(df_quarterly)
    
    # 10. 30 DMA Rule
    gann_metrics["ma_break"] = detect_gann_ma_break(df_daily, ma_period=30)
    
    # 11. 100% Rise Resistance (NEW)
    gann_metrics["hundred_percent_resistance"] = detect_100_percent_resistance(df_daily)
    
    # 12. 50% Sell Zone (NEW)
    gann_metrics["fifty_percent_sell_zone"] = detect_50_percent_sell_zone(df_daily)
    
    return gann_metrics