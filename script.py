"""
Enhanced Stock Analysis Tool for Long-Term Investors
===================================================
Comprehensive technical analysis tool optimized for long-term investment decisions.
Supports batch analysis of multiple stocks with sentiment analysis and ownership-based advice.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import os
import requests
from textblob import TextBlob
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.dates as mdates
import seaborn as sns
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

api_key = os.getenv("API_KEY")

class StockAnalyzer:
    def __init__(self, ticker: str, period: str = "1y", investment_horizon: str = "long-term", owns_stock: bool = False):
        self.ticker = ticker.upper()
        self.period = period
        self.investment_horizon = investment_horizon
        self.owns_stock = owns_stock
        self.data = None
        self.latest = None
        self.score = 0
        self.signals = []
        self.confidence_level = "Low"
        self.price_targets = {"entry": [], "exit": []}
        self.sentiment_score = 0
        self.news_headlines = []
        self.metric_info = {
            "RSI": ("Measures momentum and overbought/oversold conditions (30-70 range).",
                    ">70: Overbought (Bearish)\n<30: Oversold (Bullish)"),
            "MACD": ("Measures trend direction and momentum through moving average convergence.",
                     "Positive: Bullish trend\nNegative: Bearish trend"),
            "ADX": ("Measures trend strength regardless of direction (>25 = strong trend).",
                    ">25: Strong trend\n<20: Weak trend/Consolidation"),
            "SMA200": ("200-day Simple Moving Average - key long-term trend indicator.",
                       "Price > SMA200: Bullish\nPrice < SMA200: Bearish"),
            "Volume Ratio": ("Compares current volume to 20-day average.",
                             ">1.5: Strong conviction\n<0.5: Weak conviction"),
            "ATR": ("Measures volatility through average true range.",
                    "High: Increased risk/opportunity\nLow: Stability but limited moves"),
            "Momentum_200": ("200-day price change percentage.",
                             "Positive: Bullish momentum\nNegative: Bearish momentum")
        }
    
    def fetch_data(self) -> bool:
        try:
            ticker_obj = yf.Ticker(self.ticker)
            self.data = ticker_obj.history(period=self.period)

            if self.data.empty:
                print(f"❌ No data found for ticker: {self.ticker}")
                return False

            try:
                info = ticker_obj.info
                self.company_name = info.get('longName', self.ticker)
                self.sector = info.get('sector', 'Unknown')
                self.market_cap = info.get('marketCap', 'Unknown')
            except:
                self.company_name = self.ticker
                self.sector = 'Unknown'
                self.market_cap = 'Unknown'
            
            self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
            print(f"✅ Successfully loaded {len(self.data)} days of data for {self.ticker}")
            return True
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            return False
    
    def get_news_sentiment(self):
        """Fetch news headlines and calculate sentiment score"""
        if not api_key or api_key == "YOUR_NEWS_API_KEY":
            print("ℹ️ News API not configured. Skipping sentiment analysis.")
            return
            
        try:
            # Fetch news using NewsAPI
            url = f"https://newsapi.org/v2/everything?q={self.ticker}&apiKey={api_key}&pageSize=10"
            response = requests.get(url)
            articles = response.json().get('articles', [])
            
            sentiment_scores = []
            self.news_headlines = []
            
            for article in articles[:5]:  # Analyze top 5 articles
                title = article.get('title', '')
                content = article.get('description', '') or article.get('content', '')
                if title:
                    self.news_headlines.append(title)
                    analysis = TextBlob(title + " " + content)
                    sentiment_scores.append(analysis.sentiment.polarity)
            
            if sentiment_scores:
                self.sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
        except Exception as e:
            print(f"⚠️ Error fetching news: {str(e)}")
    
    def calculate_indicators(self):
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        volume = self.data['Volume']

        # 1. RSI
        delta = close.diff()
        gain = delta.clip(lower = 0)
        loss = -delta.clip(upper = 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain/avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # 2. Stochastic RSI
        rsi_min = self.data['RSI'].rolling(window=14).min()
        rsi_max = self.data['RSI'].rolling(window=14).max()
        self.data['StochRSI'] = (self.data['RSI'] - rsi_min) / (rsi_max - rsi_min) * 100
        
        # 3. Moving Averages
        self.data['SMA20'] = close.rolling(window=20).mean()
        self.data['SMA50'] = close.rolling(window=50).mean()
        self.data['SMA200'] = close.rolling(window=200).mean()
        self.data['EMA20'] = close.ewm(span=20).mean()
        self.data['EMA50'] = close.ewm(span=50).mean()
        self.data['EMA200'] = close.ewm(span=200).mean()
        
        # 4. Golden/Death Cross
        self.data['GC_Cross'] = np.where(
            (self.data['SMA50'] > self.data['SMA200']) & 
            (self.data['SMA50'].shift(1) <= self.data['SMA200'].shift(1)), 1, 0)
        self.data['DC_Cross'] = np.where(
            (self.data['SMA50'] < self.data['SMA200']) & 
            (self.data['SMA50'].shift(1) >= self.data['SMA200'].shift(1)), 1, 0)
        
        # 5. MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        self.data['MACD'] = ema12 - ema26
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']
        
        # 6. Bollinger Bands
        self.data['BB_Mid'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        self.data['BB_Upper'] = self.data['BB_Mid'] + 2 * bb_std
        self.data['BB_Lower'] = self.data['BB_Mid'] - 2 * bb_std
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Mid']
        
        # 7. Average True Range (ATR)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['ATR'] = true_range.rolling(window=14).mean()
        
        # 8. ADX (Average Directional Index)
        self.data['ADX'] = self._calculate_adx()
        
        # 9. Volume indicators
        self.data['Volume_SMA20'] = volume.rolling(window=20).mean()
        self.data['Volume_Ratio'] = volume / self.data['Volume_SMA20']
        
        # 10. Price momentum
        self.data['Momentum_50'] = close / close.shift(50) - 1
        self.data['Momentum_200'] = close / close.shift(200) - 1
        
        # 11. Support/Resistance levels
        self.data['Resistance'] = high.rolling(window=50).max()
        self.data['Support'] = low.rolling(window=50).min()

    def _calculate_adx(self, period: int = 14) -> pd.Series:
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the values
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx

    def analyze_signals(self):
        self.latest = self.data.iloc[-1]
        self.prev = self.data.iloc[-2]
        close = self.latest['Close']

        self.score = 0
        self.signals = []
        self.price_targets = {"entry": [], "exit": []}
        self.sentiment_score = 0  # Reset sentiment score

        trend_score = self._analyze_trend()
        self.score += trend_score

        momentum_score = self._analyze_momentum()
        self.score += momentum_score

        reversion_score = self._analyze_mean_reversion()
        self.score += reversion_score

        volume_score = self._analyze_volume()
        self.score += volume_score

        volatility_score = self._analyze_volatility()
        self.score += volatility_score

        # Adjust score based on ownership
        if self.owns_stock and self.score > 0:
            self.score -= 1  # Be more conservative for holders
        elif not self.owns_stock and self.score < 0:
            self.score += 1  # Be more conservative for non-holders

        self._calculate_price_targets()
        self._calculate_confidence()
        self.get_news_sentiment()
    
    def _analyze_trend(self) -> float:
        score = 0
        close = self.latest['Close']

        # Moving Average Alignment - Long-term focused
        if close > self.latest['SMA50'] > self.latest['SMA200']:
            if self.latest['GC_Cross']:
                score += 4
                self.signals.append("🌟 GOLDEN CROSS: Bullish long-term trend confirmed")
            else:
                score += 3
                self.signals.append("🟢 Strong uptrend: Price above major moving averages")
        elif close < self.latest['SMA50'] < self.latest['SMA200']:
            if self.latest['DC_Cross']:
                score -= 4
                self.signals.append("⚠️ DEATH CROSS: Bearish long-term trend confirmed")
            else:
                score -= 3
                self.signals.append("🔴 Strong downtrend: Price below major moving averages")
        elif abs(close - self.latest['SMA200']) < 0.05 * self.latest['SMA200']:
            self.signals.append("⚖️ Price near 200-day SMA - critical decision point")
        
        # EMA Alignment
        if close > self.latest['EMA200']:
            score += 1
            self.signals.append("📈 Price above 200-day EMA - long-term bullish")
        elif close < self.latest['EMA200']:
            score -= 1
            self.signals.append("📉 Price below 200-day EMA - long-term bearish")
        
        # ADX Trend Strength
        if self.latest['ADX'] > 30:
            if close > self.latest['SMA50']:
                score += 1.5
                self.signals.append("💪 Strong upward trend (ADX > 30)")
            else:
                score -= 1.5
                self.signals.append("⚠️ Strong downward trend (ADX > 30)")
        elif self.latest['ADX'] < 20:
            score -= 0.5
            self.signals.append("😴 Weak trend (ADX < 20) - consolidation phase")
        
        return score
    
    def _analyze_momentum(self) -> float:
        score = 0

        # RSI Analysis - Long-term perspective
        rsi = self.latest['RSI']
        rsi_prev = self.prev['RSI']
        
        if rsi < 35 and rsi > rsi_prev:
            score += 2
            self.signals.append("📈 RSI recovering from oversold (bullish reversal potential)")
        elif rsi > 65 and rsi < rsi_prev:
            score -= 2
            self.signals.append("📉 RSI declining from overbought (bearish reversal potential)")
        elif 40 < rsi < 60:
            self.signals.append("⚖️ RSI neutral - no strong momentum signal")
        
        # Stochastic RSI - Long-term confirmation
        stoch_rsi = self.latest['StochRSI']
        if stoch_rsi < 20:
            score += 1.5
            self.signals.append("🔄 Stochastic RSI oversold - potential accumulation zone")
        elif stoch_rsi > 80:
            score -= 1.5
            self.signals.append("🔄 Stochastic RSI overbought - potential distribution zone")
        
        # MACD Analysis - Long-term signals
        macd = self.latest['MACD']
        macd_signal = self.latest['MACD_Signal']
        macd_hist = self.latest['MACD_Histogram']
        macd_hist_prev = self.prev['MACD_Histogram']
        
        if macd > 0 and macd > macd_signal and macd_hist > macd_hist_prev:
            score += 2
            self.signals.append("🚀 MACD bullish & above zero - strong momentum")
        elif macd < 0 and macd < macd_signal and macd_hist < macd_hist_prev:
            score -= 2
            self.signals.append("📉 MACD bearish & below zero - weak momentum")
        
        # Long-term Price Momentum
        mom_200 = self.latest['Momentum_200']
        if mom_200 > 0.25:  # 25% gain over 200 days
            score += 2
            self.signals.append("⚡ Strong 200-day price momentum (+{:.1f}%)".format(mom_200*100))
        elif mom_200 < -0.15:  # 15% loss over 200 days
            score -= 2
            self.signals.append("⚡ Weak 200-day price momentum ({:.1f}%)".format(mom_200*100))
        
        return score

    def _analyze_mean_reversion(self) -> float:
        score = 0
        close = self.latest['Close']
        
        # Bollinger Bands - Long-term perspective
        bb_position = (close - self.latest['BB_Lower']) / (self.latest['BB_Upper'] - self.latest['BB_Lower'])
        
        if bb_position < 0.15:  # Near lower band
            score += 2
            self.signals.append("🎯 Price near lower Bollinger Band - potential LT entry")
        elif bb_position > 0.85:  # Near upper band
            score -= 2
            self.signals.append("⚠️ Price near upper Bollinger Band - potential LT exit")
        
        # Support/Resistance levels
        support_diff = (close - self.latest['Support']) / close
        resistance_diff = (self.latest['Resistance'] - close) / close
        
        if support_diff < 0.03:
            score += 1.5
            self.signals.append("🛡️ Price near strong support - potential bounce")
        elif resistance_diff < 0.03:
            score -= 1.5
            self.signals.append("⛰️ Price near strong resistance - potential pullback")
        
        return score

    def _analyze_volume(self) -> float:
        score = 0
        vol_ratio = self.latest['Volume_Ratio']
        
        if vol_ratio > 1.8:
            if self.latest['Close'] > self.latest['Open']:
                score += 1
                self.signals.append("📊 Strong bullish volume confirmation")
            else:
                score -= 1
                self.signals.append("📊 Strong bearish volume confirmation")
        elif vol_ratio < 0.5:
            score -= 0.5
            self.signals.append("📊 Low volume - lack of conviction")
        
        # Volume trend (50-day)
        vol_trend = self.data['Volume'].tail(50).mean() / self.data['Volume'].iloc[-100:-50].mean()
        if vol_trend > 1.2:
            score += 0.5
            self.signals.append("📈 Increasing volume trend - growing interest")
        elif vol_trend < 0.8:
            score -= 0.5
            self.signals.append("📉 Decreasing volume trend - waning interest")
        
        return score
    
    def _analyze_volatility(self) -> float:
        score = 0
        current_atr = self.latest['ATR']
        atr_pct = current_atr / self.latest['Close']
        avg_atr = self.data['ATR'].tail(50).mean()
        
        if atr_pct > 0.03:
            self.signals.append("⚡ High volatility - opportunity but higher risk")
        elif atr_pct < 0.01:
            self.signals.append("😴 Low volatility - stable but limited movement")
        
        # Volatility trend
        if current_atr > avg_atr * 1.3:
            self.signals.append("📈 Volatility increasing - potential trend change")
        elif current_atr < avg_atr * 0.7:
            self.signals.append("📉 Volatility decreasing - consolidation likely")
        
        return score

    def _calculate_price_targets(self):
        """Calculate potential price targets based on technical levels"""
        close = self.latest['Close']
        
        # Entry targets
        self.price_targets['entry'].append(max(self.latest['BB_Lower'], self.latest['Support']) * 0.98)
        self.price_targets['entry'].append(self.latest['SMA200'] * 0.98)
        
        # Exit targets
        self.price_targets['exit'].append(min(self.latest['BB_Upper'], self.latest['Resistance']) * 1.02)
        self.price_targets['exit'].append(self.latest['SMA200'] * 1.20)
        
        # Remove duplicates and sort
        self.price_targets['entry'] = sorted(set(self.price_targets['entry']))
        self.price_targets['exit'] = sorted(set(self.price_targets['exit']))
        
        # Narrow ranges for HOLD recommendations
        if self.get_recommendation().startswith("HOLD"):
            self.price_targets['entry'] = [min(self.price_targets['entry']), max(self.price_targets['entry'])]
            self.price_targets['exit'] = [min(self.price_targets['exit']), max(self.price_targets['exit'])]

    def _calculate_confidence(self):
        """Calculate confidence level based on signal strength"""
        strong_signals = len([s for s in self.signals if '🌟' in s or '🚀' in s or '💀' in s])
        
        if abs(self.score) >= 8 and strong_signals >= 3:
            self.confidence_level = "High"
        elif abs(self.score) >= 5 and strong_signals >= 2:
            self.confidence_level = "Medium"
        else:
            self.confidence_level = "Low"
    
    def get_recommendation(self) -> str:
        """Generate final recommendation with trend context and ownership"""
        base_rec = ""
        context = ""
        
        if self.score >= 8:
            base_rec = "STRONG BUY"
            context = "(Early Trend)" if self.latest['ADX'] < 25 else "(Trend Established)"
        elif self.score >= 5:
            base_rec = "BUY"
            context = "(Accumulation)"
        elif self.score <= -8:
            base_rec = "STRONG SELL"
            context = "(Downtrend)"
        elif self.score <= -5:
            base_rec = "SELL"
            context = "(Distribution)"
        else:
            base_rec = "HOLD"
            context = "(Neutral)"
        
        # Adjust recommendation based on ownership
        if self.owns_stock:
            if "BUY" in base_rec:
                return "HOLD" + " " + context + " - Consider adding on dips"
            elif "SELL" in base_rec:
                return base_rec + " " + context + " - Consider reducing position"
        
        return base_rec + " " + context

    def get_recommendation_emoji(self) -> str:
        rec = self.get_recommendation()
        emoji_map = {
            "STRONG BUY": "🚀🌱",
            "STRONG BUY (Early Trend)": "🚀🌱",
            "STRONG BUY (Trend Established)": "🚀📈",
            "BUY (Accumulation)": "🟢📊",
            "HOLD (Neutral)": "🟡⚖️",
            "HOLD (Neutral) - Consider adding on dips": "🟡⬆️",
            "SELL (Distribution)": "🔴📉",
            "SELL (Distribution) - Consider reducing position": "🔴⬇️",
            "STRONG SELL (Downtrend)": "💀📉"
        }
        return emoji_map.get(rec, "🟡⚖️")

    def print_analysis(self):
        """Print comprehensive analysis results"""
        rec = self.get_recommendation()
        emoji = self.get_recommendation_emoji()
        
        print("\n" + "=" * 80)
        print(f"📊 LONG-TERM STOCK ANALYSIS REPORT")
        print("=" * 80)
        print(f"🏢 Company: {self.company_name}")
        print(f"📈 Ticker: {self.ticker} | Sector: {self.sector}")
        print(f"👤 Ownership: {'Yes' if self.owns_stock else 'No'}")
        print(f"⏱️ Analysis Period: {self.period.upper()} | Horizon: {self.investment_horizon.title()}")
        print(f"💰 Current Price: ${self.latest['Close']:.2f}")
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("\n" + "🌟" * 40)
        
        print(f"🎯 RECOMMENDATION: {emoji} {rec}")
        print(f"📊 Technical Score: {self.score:.1f}/10")
        print(f"📰 Sentiment Score: {self.sentiment_score:.2f} (Range: -1 to 1)")
        print(f"🎯 Confidence Level: {self.confidence_level}")
        print("\n" + "-" * 80)
        
        print("📋 KEY SIGNALS:")
        print("-" * 40)
        if self.signals:
            for i, signal in enumerate(self.signals, 1):
                print(f"{i:2d}. {signal}")
        else:
            print("No strong signals detected.")
        
        # Highlight price targets for HOLD recommendations
        if rec.startswith("HOLD"):
            print("\n🎯 STRATEGIC PRICE TARGETS FOR HOLD POSITION:")
            print("-" * 60)
            print("💡 For current holders, consider these strategic price levels:")
            print(f"• Ideal ADDING zone: ${min(self.price_targets['entry']):.2f} - ${max(self.price_targets['entry']):.2f}")
            print(f"• Ideal PROFIT-TAKING zone: ${min(self.price_targets['exit']):.2f} - ${max(self.price_targets['exit']):.2f}")
            print(f"• Risk/Reward Ratio: 1:{((min(self.price_targets['exit']) - self.latest['Close']) / (self.latest['Close'] - min(self.price_targets['entry']))):.1f}")
        else:
            print("\n🎯 PRICE TARGETS (Technical Estimates):")
            print("-" * 40)
            print(f"• Potential Entry Zones: ${min(self.price_targets['entry']):.2f} - ${max(self.price_targets['entry']):.2f}")
            print(f"• Potential Exit Targets: ${min(self.price_targets['exit']):.2f} - ${max(self.price_targets['exit']):.2f}")
        
        # News and sentiment section
        if self.news_headlines:
            print("\n📰 LATEST NEWS & SENTIMENT:")
            print("-" * 40)
            print(f"Sentiment Score: {self.sentiment_score:.2f} ({self._get_sentiment_label()})")
            print("Top Headlines:")
            for i, headline in enumerate(self.news_headlines[:3], 1):
                print(f"{i}. {headline}")
        
        print("\n💡 INVESTMENT GUIDANCE:")
        print("-" * 40)
        self._print_investment_guidance(rec)
        
        print("\n📊 KEY METRICS WITH EXPLANATION:")
        print("-" * 40)
        self._print_key_metrics()
        
        print("\n💎 LONG-TERM INSIGHTS:")
        print("-" * 40)
        self._print_long_term_insights()
        
        print("\n" + "=" * 80)
    
    def _get_sentiment_label(self):
        """Get sentiment label based on score"""
        if self.sentiment_score > 0.3:
            return "Strongly Positive"
        elif self.sentiment_score > 0.1:
            return "Positive"
        elif self.sentiment_score < -0.3:
            return "Strongly Negative"
        elif self.sentiment_score < -0.1:
            return "Negative"
        return "Neutral"
    
    def _print_investment_guidance(self, recommendation: str):
        guidance = {
            "STRONG BUY (Early Trend)": [
                "Strong accumulation signals detected",
                "Ideal for dollar-cost averaging into position",
                "Set stop-loss 8-10% below entry price",
                "Monitor for trend confirmation signals"
            ],
            "STRONG BUY (Trend Established)": [
                "Strong trend with momentum confirmation",
                "Consider initiating core position",
                "Set trailing stop-loss 10-15% below current price",
                "Rebalance at key resistance levels"
            ],
            "BUY (Accumulation)": [
                "Favorable risk/reward profile",
                "Consider scaling into position",
                "Set stop-loss 7-9% below entry price",
                "Watch volume on breakout attempts"
            ],
            "HOLD (Neutral)": [
                "Market in consolidation phase",
                "Maintain current position size",
                "Consider rebalancing at range extremes",
                "Monitor for breakout/breakdown signals"
            ],
            "HOLD (Neutral) - Consider adding on dips": [
                "Technical signals mixed but long-term outlook positive",
                "Maintain core position",
                "Consider adding at strategic support levels",
                "Set stop-loss below key support"
            ],
            "SELL (Distribution)": [
                "Technical deterioration evident",
                "Consider reducing position size",
                "Implement tighter stop-loss management",
                "Rebalance into stronger opportunities"
            ],
            "SELL (Distribution) - Consider reducing position": [
                "Bearish signals emerging",
                "Reduce exposure to preserve capital",
                "Set stop-loss above recent highs",
                "Consider hedging strategies"
            ],
            "STRONG SELL (Downtrend)": [
                "Strong bearish momentum confirmed",
                "Consider exiting core positions",
                "Preserve capital for future opportunities",
                "Wait for clear reversal signals before re-entry"
            ]
        }
        
        for point in guidance.get(recommendation, []):
            print(f"• {point}")
        
        # Add general long-term advice
        print("\n💎 Long-Term Strategy:")
        print("• Maintain minimum 6-12 month investment horizon")
        print("• Rebalance portfolio quarterly")
        print("• Always use position sizing appropriate to your risk tolerance")
        
    def _print_key_metrics(self):
        metrics = [
            ("RSI (14)", f"{self.latest['RSI']:.1f}"),
            ("MACD", f"{self.latest['MACD']:.3f}"),
            ("ADX (Trend Strength)", f"{self.latest['ADX']:.1f}"),
            ("200-day Momentum", f"{self.latest['Momentum_200']*100:.1f}%"),
            ("Price/200SMA", f"{(self.latest['Close']/self.latest['SMA200']-1)*100:.1f}%"),
            ("Volume Ratio", f"{self.latest['Volume_Ratio']:.1f}x"),
            ("ATR (Volatility)", f"{self.latest['ATR']:.2f} ({self.latest['ATR']/self.latest['Close']*100:.1f}%)")
        ]
        
        for name, value in metrics:
            info = self.metric_info.get(name.split(' ')[0], ("", ""))
            print(f"{name}: {value}")
            print(f"   • What it measures: {info[0]}")
            print(f"   • Interpretation: {info[1]}")
            print()
    
    def _print_long_term_insights(self):
        """Provide long-term insights based on analysis"""
        insights = []
        
        # Trend insights
        if self.latest['ADX'] > 25:
            if self.latest['Close'] > self.latest['SMA200']:
                insights.append("Strong uptrend in place - favorable for long-term positions")
            else:
                insights.append("Strong downtrend - consider defensive positioning")
        else:
            insights.append("Market in consolidation phase - accumulation opportunity")
        
        # Momentum insights
        rsi = self.latest['RSI']
        if rsi < 40:
            insights.append("Oversold conditions present - potential accumulation opportunity")
        elif rsi > 60:
            insights.append("Approaching overbought territory - monitor for profit-taking opportunities")
        
        # Volatility insights
        atr_pct = self.latest['ATR'] / self.latest['Close']
        if atr_pct > 0.03:
            insights.append("Elevated volatility - position size accordingly")
        else:
            insights.append("Low volatility environment - suitable for core positions")
        
        # Ownership-specific insights
        if self.owns_stock:
            insights.append("As a current holder, focus on risk management and position sizing")
        else:
            insights.append("As a potential buyer, look for strategic entry points in the target range")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")

    def plot_analysis(self, save_plot: bool = False):
        """Create clean, focused analysis plots"""
        sns.set_style('whitegrid')
        fig, axes = plt.subplots(3, 1, figsize=(15, 14))
        fig.suptitle(f'{self.ticker} - Long-Term Analysis ({self.period.upper()})', 
                    fontsize=16, fontweight='bold')
        
        # Configure date formatting
        date_fmt = mdates.DateFormatter('%b %Y')
        
        # Plot 1: Price and Moving Averages (Clean)
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], label='Price', linewidth=2, color='#1f77b4')
        ax1.plot(self.data.index, self.data['SMA200'], label='SMA 200', linewidth=1.5, color='#ff7f0e', linestyle='-')
        ax1.plot(self.data.index, self.data['EMA200'], label='EMA 200', linewidth=1.5, color='#2ca02c', linestyle='--')
        
        # Highlight Golden/Death Crosses
        gc_dates = self.data[self.data['GC_Cross'] == 1].index
        dc_dates = self.data[self.data['DC_Cross'] == 1].index
        
        for date in gc_dates:
            ax1.axvline(x=date, color='green', alpha=0.4, linestyle='--', linewidth=1)
        for date in dc_dates:
            ax1.axvline(x=date, color='red', alpha=0.4, linestyle='--', linewidth=1)
        
        # Add support/resistance levels
        ax1.plot(self.data.index, self.data['Support'], alpha=0.5, color='blue', linestyle=':', label='Support')
        ax1.plot(self.data.index, self.data['Resistance'], alpha=0.5, color='purple', linestyle=':', label='Resistance')
        
        ax1.set_title('Price Action with Key Levels', fontsize=14)
        ax1.legend(loc='best', fontsize=9)
        ax1.xaxis.set_major_formatter(date_fmt)
        
        # Plot 2: Momentum Indicators
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['RSI'], label='RSI', color='#d62728', linewidth=1.5)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.fill_between(self.data.index, 30, 70, alpha=0.1, color='gray')
        
        # MACD with histogram
        ax2_macd = ax2.twinx()
        ax2_macd.plot(self.data.index, self.data['MACD'], label='MACD', color='#17becf', linewidth=1.5, alpha=0.7)
        ax2_macd.plot(self.data.index, self.data['MACD_Signal'], label='Signal', color='#e377c2', linewidth=1.5, alpha=0.7)
        ax2_macd.bar(self.data.index, self.data['MACD_Histogram'], 
                     color=np.where(self.data['MACD_Histogram'] >= 0, '#4daf4a', '#e41a1c'), 
                     alpha=0.5, label='Histogram')
        
        ax2.set_title('Momentum Indicators', fontsize=14)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left', fontsize=9)
        ax2_macd.legend(loc='upper right', fontsize=9)
        ax2.xaxis.set_major_formatter(date_fmt)
        
        # Plot 3: Volume and ADX
        ax3 = axes[2]
        # Volume bars
        ax3.bar(self.data.index, self.data['Volume'], color='#aec7e8', alpha=0.6, label='Volume')
        ax3.plot(self.data.index, self.data['Volume_SMA20'], color='#1f77b4', linewidth=1.5, label='20-day Avg Volume')
        
        # ADX on secondary axis
        ax3_adx = ax3.twinx()
        ax3_adx.plot(self.data.index, self.data['ADX'], color='#9467bd', linewidth=1.5, label='ADX (Trend Strength)')
        ax3_adx.axhline(y=25, color='#8c564b', linestyle='--', alpha=0.7, label='Trend Threshold')
        
        ax3.set_title('Volume & Trend Strength', fontsize=14)
        ax3.legend(loc='upper left', fontsize=9)
        ax3_adx.legend(loc='upper right', fontsize=9)
        ax3.xaxis.set_major_formatter(date_fmt)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_plot:
            filename = f"{self.ticker}_LT_analysis_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Chart saved as: {filename}")
        
        plt.show()

    def save_report(self, filename: Optional[str] = None):
        """Save analysis report to CSV"""
        if filename is None:
            filename = f"{self.ticker}_LT_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        # Create summary data
        summary_data = {
            'Ticker': [self.ticker],
            'Company': [self.company_name],
            'Owns_Stock': [self.owns_stock],
            'Analysis_Date': [datetime.now().strftime('%Y-%m-%d %H:%M')],
            'Recommendation': [self.get_recommendation()],
            'Technical_Score': [self.score],
            'Sentiment_Score': [self.sentiment_score],
            'Confidence': [self.confidence_level],
            'Current_Price': [self.latest['Close']],
            'RSI': [self.latest['RSI']],
            'MACD': [self.latest['MACD']],
            'ADX': [self.latest['ADX']],
            'Volume_Ratio': [self.latest['Volume_Ratio']],
            'Momentum_200': [self.latest['Momentum_200']],
            'Price_SMA200_Ratio': [self.latest['Close']/self.latest['SMA200']],
            'Entry_Targets': [' - '.join(f"{x:.2f}" for x in self.price_targets['entry'])],
            'Exit_Targets': [' - '.join(f"{x:.2f}" for x in self.price_targets['exit'])],
            'Signals': [' | '.join(self.signals)]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filename, index=False)
        print(f"💾 Report saved as: {filename}")
    
def main():
    """Main function with enhanced user interface"""
    print("\n" + "=" * 70)
    print("🌟 ENHANCED STOCK ANALYSIS TOOL FOR LONG-TERM INVESTORS")
    print("=" * 70)
    print("📈 Optimized for Strategic Investing (3+ month horizon)")
    print("💼 Combines Technical Analysis with News Sentiment")
    print("\n" + "-" * 70)
    
    # Analysis configuration
    period_map = {'1': '6mo', '2': '1y', '3': '2y', '4': '5y', '5': 'max'}
    print("\n📅 Select time period:")
    print("1. 6 months (6mo)")
    print("2. 1 year (1y) - Recommended")
    print("3. 2 years (2y)")
    print("4. 5 years (5y)")
    print("5. Max available")
    period_choice = input("Choose period (1-5): ").strip()
    period = period_map.get(period_choice, '1y')
    
    print("\n🔭 Analysis Options:")
    show_plot = input("Show technical charts? (y/n): ").strip().lower() == 'y'
    save_plot = input("Save charts? (y/n): ").strip().lower() == 'y' if show_plot else False
    save_report = input("Save analysis reports? (y/n): ").strip().lower() == 'y'
    
    # Multi-stock analysis loop
    while True:
        print("\n" + "=" * 70)
        ticker = input("\n📊 Enter stock ticker (or 'done' to finish): ").strip().upper()
        
        if ticker == 'DONE':
            break
            
        if not ticker or len(ticker) > 5:
            print("❌ Invalid ticker symbol")
            continue
        
        # Ownership question
        owns_stock = input(f"Do you currently own {ticker}? (y/n): ").strip().lower() == 'y'
            
        print(f"\n🔍 Analyzing {ticker}... Please wait...")
        
        # Create analyzer and run analysis
        analyzer = StockAnalyzer(ticker, period, "long-term", owns_stock)
        
        if not analyzer.fetch_data():
            continue
        
        analyzer.calculate_indicators()
        analyzer.analyze_signals()
        analyzer.print_analysis()
        
        if show_plot:
            analyzer.plot_analysis(save_plot=save_plot)
        
        if save_report:
            analyzer.save_report()
    
    # Final output and suggestions
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("💡 Remember: Technical analysis is one tool - always consider fundamentals")
    print("\n🔮 FUTURE ENHANCEMENT SUGGESTIONS:")
    print("-" * 70)
    print("1. Fundamental Analysis Integration:")
    print("   • Add P/E, P/B, and dividend yield metrics")
    print("   • Incorporate earnings growth projections")
    print("   • Analyze debt-to-equity ratios")
    
    print("\n2. Advanced Analytics:")
    print("   • Portfolio optimization suggestions")
    print("   • Correlation analysis between stocks")
    print("   • Risk-adjusted return metrics (Sharpe ratio)")
    
    print("\n3. User Experience Improvements:")
    print("   • Interactive web-based dashboard")
    print("   • Mobile app with push notifications")
    print("   • Custom alert system for price targets")
    
    print("\n4. Data Source Expansion:")
    print("   • Economic indicators (interest rates, inflation)")
    print("   • Sector performance comparisons")
    print("   • Insider trading activity")
    
    print("\n5. Machine Learning Features:")
    print("   • Price prediction models")
    print("   • Anomaly detection for unusual activity")
    print("   • Sentiment analysis of earnings calls")
    
    print("\n" + "=" * 70)
    print("📈 Happy Investing! 💼💰")


if __name__ == "__main__":
    main()