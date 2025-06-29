"""
Enhanced Stock Analysis Tool for Long-Term Investors
===================================================
Comprehensive technical analysis tool optimized for long-term investment decisions.
Combines technical indicators, fundamental analysis, and news sentiment.
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
import time
import random
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
        self.analyst_data = {}
        self.fundamental_data = {}
        
        # Comprehensive metric explanations
        self.metric_info = {
            "RSI": {
                "measure": "Measures momentum and overbought/oversold conditions on a scale of 0-100",
                "interpretation": "<30: Oversold (potential buying opportunity)\n30-70: Neutral range\n>70: Overbought (potential selling opportunity)",
                "long_term": "More reliable in trending markets than sideways markets"
            },
            "MACD": {
                "measure": "Measures trend direction and momentum through moving average convergence/divergence",
                "interpretation": "Positive MACD: Bullish momentum\nNegative MACD: Bearish momentum\nCrossovers signal trend changes",
                "long_term": "MACD crossovers provide stronger signals on weekly/monthly charts"
            },
            "ADX": {
                "measure": "Measures trend strength regardless of direction (0-100 scale)",
                "interpretation": "<20: Weak trend/consolidation\n20-25: Emerging trend\n>25: Strong trend",
                "long_term": "Best used to confirm whether a trend is worth trading"
            },
            "SMA200": {
                "measure": "200-day Simple Moving Average - key long-term trend indicator",
                "interpretation": "Price > SMA200: Bullish long-term trend\nPrice < SMA200: Bearish long-term trend",
                "long_term": "The most important moving average for long-term investors"
            },
            "Volume": {
                "measure": "Trading volume relative to 20-day average",
                "interpretation": ">1.5: High conviction move\n<0.5: Low conviction move",
                "long_term": "Volume spikes often precede significant price moves"
            },
            "ATR": {
                "measure": "Average True Range - measures volatility based on price range",
                "interpretation": "High ATR: Increased volatility/risk\nLow ATR: Reduced volatility/stability",
                "long_term": "Helps determine position sizing and stop-loss levels"
            },
            "Momentum": {
                "measure": "200-day price change percentage",
                "interpretation": "Positive: Bullish momentum\nNegative: Bearish momentum",
                "long_term": "Shows long-term price direction strength"
            },
            "P/E": {
                "measure": "Price-to-Earnings ratio - valuation metric",
                "interpretation": "Lower than industry: Potentially undervalued\nHigher than industry: Potentially overvalued",
                "long_term": "Most reliable when compared to historical P/E and sector average"
            },
            "EPS Growth": {
                "measure": "Earnings Per Share growth rate - company profitability trend",
                "interpretation": ">15%: Strong growth\n<5%: Weak growth",
                "long_term": "Sustained growth is key for long-term stock appreciation"
            },
            "Dividend Yield": {
                "measure": "Annual dividend payment as percentage of stock price",
                "interpretation": ">4%: High income potential\n<2%: Low income potential",
                "long_term": "Important for income-focused investors in retirement"
            }
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
                self.industry = info.get('industry', 'Unknown')
                self.market_cap = info.get('marketCap', 'Unknown')
                
                # Fundamental data
                self.fundamental_data = {
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'eps_growth': info.get('earningsQuarterlyGrowth'),
                    'dividend_yield': info.get('dividendYield'),
                    'profit_margins': info.get('profitMargins'),
                    'roe': info.get('returnOnEquity'),
                    'debt_to_equity': info.get('debtToEquity')
                }
                
                # Analyst data
                self.analyst_data = {
                    'recommendation': info.get('recommendationMean'),
                    'target_low': info.get('targetLowPrice'),
                    'target_med': info.get('targetMedianPrice'),
                    'target_high': info.get('targetHighPrice'),
                    'number_of_analysts': info.get('numberOfAnalystOpinions')
                }
                
            except Exception as e:
                print(f"⚠️ Warning: {str(e)}")
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
            url = f"https://newsapi.org/v2/everything?q={self.ticker}&apiKey={api_key}&pageSize=5&language=en"
            response = requests.get(url)
            articles = response.json().get('articles', [])
            
            sentiment_scores = []
            self.news_headlines = []
            
            for article in articles:
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
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # 2. Stochastic RSI (long-term optimized)
        rsi_min = self.data['RSI'].rolling(window=21).min()
        rsi_max = self.data['RSI'].rolling(window=21).max()
        self.data['StochRSI'] = (self.data['RSI'] - rsi_min) / (rsi_max - rsi_min) * 100
        
        # 3. Moving Averages
        self.data['SMA50'] = close.rolling(window=50).mean()
        self.data['SMA200'] = close.rolling(window=200).mean()
        self.data['EMA200'] = close.ewm(span=200).mean()
        
        # 4. Golden/Death Cross
        self.data['GC_Cross'] = np.where(
            (self.data['SMA50'] > self.data['SMA200']) & 
            (self.data['SMA50'].shift(1) <= self.data['SMA200'].shift(1)), 1, 0)
        self.data['DC_Cross'] = np.where(
            (self.data['SMA50'] < self.data['SMA200']) & 
            (self.data['SMA50'].shift(1) >= self.data['SMA200'].shift(1)), 1, 0)
        
        # 5. MACD (long-term optimized)
        ema26 = close.ewm(span=26).mean()
        ema52 = close.ewm(span=52).mean()
        self.data['MACD'] = ema26 - ema52
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9).mean()
        
        # 6. Bollinger Bands
        self.data['BB_Mid'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        self.data['BB_Upper'] = self.data['BB_Mid'] + 2 * bb_std
        self.data['BB_Lower'] = self.data['BB_Mid'] - 2 * bb_std
        
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
        
        # 10. Long-term Momentum
        self.data['Momentum_200'] = (close / close.shift(200) - 1) * 100

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
        close = self.latest['Close']

        self.score = 0
        self.signals = []
        self.price_targets = {"entry": [], "exit": []}

        # Technical analysis
        self.score += self._analyze_trend()
        self.score += self._analyze_momentum()
        self.score += self._analyze_mean_reversion()
        self.score += self._analyze_volume()
        self.score += self._analyze_volatility()
        
        # Fundamental analysis
        fundamental_score = self._analyze_fundamentals()
        self.score += fundamental_score
        
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
        
        # Price position relative to 200-day SMA
        sma200_dist = (close - self.latest['SMA200']) / self.latest['SMA200']
        if abs(sma200_dist) < 0.05:
            score += 0.5 if sma200_dist > 0 else -0.5
            self.signals.append("⚖️ Price near 200-day SMA - critical decision point")
        
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
        if rsi < 35:
            score += 1.5
            self.signals.append("📈 RSI oversold - potential accumulation zone")
        elif rsi > 65:
            score -= 1.5
            self.signals.append("📉 RSI overbought - potential distribution zone")
        
        # MACD Analysis - Long-term signals
        macd = self.latest['MACD']
        macd_signal = self.latest['MACD_Signal']
        
        if macd > 0 and macd > macd_signal:
            score += 1.5
            self.signals.append("🚀 MACD bullish - positive momentum")
        
        # Long-term Price Momentum
        mom_200 = self.latest['Momentum_200']
        if mom_200 > 25:  # 25% gain over 200 days
            score += 2
            self.signals.append(f"⚡ Strong 200-day price momentum (+{mom_200:.1f}%)")
        elif mom_200 < -15:  # 15% loss over 200 days
            score -= 2
            self.signals.append(f"⚡ Weak 200-day price momentum ({mom_200:.1f}%)")
        
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
        support_diff = (close - self.latest['BB_Lower']) / close
        resistance_diff = (self.latest['BB_Upper'] - close) / close
        
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
            score += 1
            self.signals.append("📊 Strong volume confirmation - high conviction move")
        elif vol_ratio < 0.5:
            score -= 0.5
            self.signals.append("📊 Low volume - lack of conviction")
        
        return score
    
    def _analyze_volatility(self) -> float:
        current_atr = self.latest['ATR']
        atr_pct = current_atr / self.latest['Close'] * 100
        
        if atr_pct > 3:
            self.signals.append("⚡ High volatility ({:.1f}%) - higher risk/opportunity".format(atr_pct))
        elif atr_pct < 1:
            self.signals.append("😴 Low volatility ({:.1f}%) - stable but limited moves".format(atr_pct))
        
        return 0
    
    def _analyze_fundamentals(self) -> float:
        """Analyze fundamental data for long-term investment"""
        score = 0
        pe = self.fundamental_data.get('pe_ratio')
        eps_growth = self.fundamental_data.get('eps_growth')
        div_yield = self.fundamental_data.get('dividend_yield')
        
        if pe and pe < 20:
            score += 1
            self.signals.append("💰 Reasonable P/E ratio: {:.1f}".format(pe))
        elif pe and pe > 30:
            score -= 1
            self.signals.append("⚠️ High P/E ratio: {:.1f} - valuation concern".format(pe))
            
        if eps_growth and eps_growth > 0.15:
            score += 1.5
            self.signals.append("📈 Strong EPS growth: {:.1%}".format(eps_growth))
        elif eps_growth and eps_growth < 0.05:
            score -= 1
            self.signals.append("📉 Weak EPS growth: {:.1%}".format(eps_growth))
            
        if div_yield and div_yield > 0.04:
            score += 1
            self.signals.append("💵 Attractive dividend yield: {:.2%}".format(div_yield))
            
        return score

    def _calculate_price_targets(self):
        """Calculate potential price targets based on technical levels"""
        close = self.latest['Close']
        
        # Entry targets
        self.price_targets['entry'].append(max(self.latest['BB_Lower'], self.latest['SMA200'] * 0.95) * 0.98)
        self.price_targets['entry'].append(self.latest['SMA200'] * 0.98)
        
        # Exit targets
        self.price_targets['exit'].append(min(self.latest['BB_Upper'], self.latest['Resistance']) * 1.02)
        self.price_targets['exit'].append(self.latest['SMA200'] * 1.20)
        
        # Analyst targets if available
        if self.analyst_data.get('target_med'):
            self.price_targets['exit'].append(self.analyst_data['target_med'])
        
        # Remove duplicates and sort
        self.price_targets['entry'] = sorted(set(self.price_targets['entry']))
        self.price_targets['exit'] = sorted(set(self.price_targets['exit']))
        
        # Narrow ranges for HOLD recommendations
        if self.get_recommendation().startswith("HOLD"):
            # Create tighter range for clearer decision-making
            self.price_targets['entry'] = [min(self.price_targets['entry']), 
                                          min(self.price_targets['entry']) * 1.05]
            self.price_targets['exit'] = [max(self.price_targets['exit']) * 0.95, 
                                         max(self.price_targets['exit'])]

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
        sentiment_label = self._get_sentiment_label()
        
        print("\n" + "=" * 80)
        print(f"📊 LONG-TERM STOCK ANALYSIS REPORT")
        print("=" * 80)
        print(f"🏢 Company: {self.company_name}")
        print(f"📈 Ticker: {self.ticker} | Sector: {self.sector}")
        print(f"👤 Ownership: {'Yes' if self.owns_stock else 'No'}")
        print(f"⏱️ Analysis Period: {self.period.upper()} | Horizon: Long-term investment")
        print(f"💰 Current Price: ${self.latest['Close']:.2f}")
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("\n" + "🌟" * 40)
        
        print(f"🎯 RECOMMENDATION: {emoji} {rec}")
        print(f"📊 Technical Score: {self.score:.1f}/10")
        print(f"📰 Sentiment: {sentiment_label} ({self.sentiment_score:.2f})")
        print(f"🎯 Confidence Level: {self.confidence_level}")
        print("\n" + "-" * 80)
        
        print("📋 KEY SIGNALS:")
        print("-" * 40)
        if self.signals:
            for i, signal in enumerate(self.signals, 1):
                print(f"{i:2d}. {signal}")
        else:
            print("No strong signals detected.")
        
        # Price targets section
        print("\n🎯 PRICE TARGETS:")
        print("-" * 40)
        
        # Special section for HOLD recommendations
        if rec.startswith("HOLD"):
            print("💡 For current holders, consider these strategic price levels:")
            print(f"• Ideal ADDING zone: ${self.price_targets['entry'][0]:.2f} - ${self.price_targets['entry'][1]:.2f}")
            print(f"• Ideal PROFIT-TAKING zone: ${self.price_targets['exit'][0]:.2f} - ${self.price_targets['exit'][1]:.2f}")
            print(f"• Risk/Reward Ratio: 1:{((min(self.price_targets['exit']) - self.latest['Close']) / (self.latest['Close'] - min(self.price_targets['entry']))):.1f}")
        else:
            print(f"• Potential Entry Zones: ${self.price_targets['entry'][0]:.2f} - ${self.price_targets['entry'][1]:.2f}")
            print(f"• Potential Exit Targets: ${self.price_targets['exit'][0]:.2f} - ${self.price_targets['exit'][1]:.2f}")
        
        # Analyst targets if available
        if self.analyst_data.get('target_med'):
            print("\n📈 ANALYST CONSENSUS:")
            print("-" * 40)
            print(f"• Recommendation: {self._get_analyst_recommendation()}")
            print(f"• Target Prices: Low ${self.analyst_data['target_low']:.2f} | "
                  f"Med ${self.analyst_data['target_med']:.2f} | "
                  f"High ${self.analyst_data['target_high']:.2f}")
            print(f"• Based on {self.analyst_data['number_of_analysts']} analysts")
        
        # News and sentiment section
        if self.news_headlines:
            print("\n📰 LATEST NEWS & SENTIMENT:")
            print("-" * 40)
            print(f"Sentiment: {sentiment_label} ({self.sentiment_score:.2f})")
            print("Top Headlines:")
            for i, headline in enumerate(self.news_headlines[:3], 1):
                print(f"{i}. {headline}")
        
        print("\n📊 KEY METRICS WITH EXPLANATION:")
        print("-" * 40)
        self._print_key_metrics()
        
        print("\n💎 LONG-TERM INSIGHTS:")
        print("-" * 40)
        self._print_long_term_insights()
        
        print("\n" + "=" * 80)
    
    def _get_sentiment_label(self):
        """Get plain language sentiment label"""
        if self.sentiment_score > 0.3:
            return "Strongly Positive"
        elif self.sentiment_score > 0.1:
            return "Positive"
        elif self.sentiment_score < -0.3:
            return "Strongly Negative"
        elif self.sentiment_score < -0.1:
            return "Negative"
        return "Neutral"
    
    def _get_analyst_recommendation(self):
        """Convert analyst mean recommendation to text"""
        rating = self.analyst_data.get('recommendation')
        if not rating:
            return "No data"
        
        if rating <= 1.5:
            return "Strong Buy"
        elif rating <= 2.5:
            return "Buy"
        elif rating <= 3.5:
            return "Hold"
        elif rating <= 4.5:
            return "Sell"
        return "Strong Sell"
    
    def _print_key_metrics(self):
        metrics = [
            ("RSI (14)", f"{self.latest['RSI']:.1f}", "RSI"),
            ("MACD", f"{self.latest['MACD']:.3f}", "MACD"),
            ("ADX (Trend Strength)", f"{self.latest['ADX']:.1f}", "ADX"),
            ("200-day Momentum", f"{self.latest['Momentum_200']:.1f}%", "Momentum"),
            ("Price/200SMA", f"{(self.latest['Close']/self.latest['SMA200']-1)*100:.1f}%", "SMA200"),
            ("Volume Ratio", f"{self.latest['Volume_Ratio']:.1f}x", "Volume"),
            ("ATR (Volatility)", f"{self.latest['ATR']:.2f} ({self.latest['ATR']/self.latest['Close']*100:.1f}%)", "ATR")
        ]
        
        # Add fundamental metrics if available
        if self.fundamental_data.get('pe_ratio'):
            metrics.append(("P/E Ratio", f"{self.fundamental_data['pe_ratio']:.1f}", "P/E"))
        if self.fundamental_data.get('eps_growth'):
            metrics.append(("EPS Growth", f"{self.fundamental_data['eps_growth']*100:.1f}%", "EPS Growth"))
        if self.fundamental_data.get('dividend_yield'):
            metrics.append(("Dividend Yield", f"{self.fundamental_data['dividend_yield']*100:.2f}%", "Dividend Yield"))
        
        for name, value, key in metrics:
            info = self.metric_info.get(key, {"measure": "", "interpretation": ""})
            print(f"{name}: {value}")
            print(f"   • What it measures: {info.get('measure', 'Not available')}")
            print(f"   • Interpretation: {info.get('interpretation', 'Not available')}")
            print(f"   • Long-term perspective: {info.get('long_term', 'Not available')}")
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
        
        # Fundamental insights
        if self.fundamental_data.get('pe_ratio') and self.fundamental_data['pe_ratio'] < 15:
            insights.append("Attractive valuation based on P/E ratio")
        if self.fundamental_data.get('eps_growth') and self.fundamental_data['eps_growth'] > 0.2:
            insights.append("Strong earnings growth supports long-term appreciation")
        
        # Ownership-specific insights
        if self.owns_stock:
            insights.append("As a current holder, focus on risk management and position sizing")
            insights.append("Consider rebalancing if this position exceeds 5% of your portfolio")
        else:
            insights.append("As a potential buyer, look for strategic entry points in the target range")
            insights.append("Consider dollar-cost averaging to reduce timing risk")
        
        # Sentiment insights
        sentiment = self._get_sentiment_label()
        if sentiment == "Strongly Positive":
            insights.append("Very positive market sentiment - confirms bullish case")
        elif sentiment == "Negative":
            insights.append("Negative market sentiment - warrants caution")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")

    def plot_analysis(self, save_plot: bool = False):
        """Create clean, focused analysis plots for long-term investors"""
        sns.set_style('whitegrid')
        fig, axes = plt.subplots(3, 1, figsize=(15, 14))
        fig.suptitle(f'{self.ticker} - Long-Term Analysis ({self.period.upper()})', 
                    fontsize=16, fontweight='bold')
        
        # Configure date formatting
        date_fmt = mdates.DateFormatter('%b %Y')
        
        # Plot 1: Price and Moving Averages (Clean)
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], label='Price', linewidth=2, color='#1f77b4')
        ax1.plot(self.data.index, self.data['SMA200'], label='200-day SMA', linewidth=1.5, color='#ff7f0e')
        
        # Highlight Golden/Death Crosses
        gc_dates = self.data[self.data['GC_Cross'] == 1].index
        dc_dates = self.data[self.data['DC_Cross'] == 1].index
        
        for date in gc_dates:
            ax1.axvline(x=date, color='green', alpha=0.4, linestyle='--', linewidth=1)
        for date in dc_dates:
            ax1.axvline(x=date, color='red', alpha=0.4, linestyle='--', linewidth=1)
        
        # Add support/resistance levels
        ax1.plot(self.data.index, self.data['BB_Lower'], alpha=0.5, color='blue', linestyle=':', label='Support')
        ax1.plot(self.data.index, self.data['BB_Upper'], alpha=0.5, color='purple', linestyle=':', label='Resistance')
        
        ax1.set_title('Price Action with Key Levels', fontsize=14)
        ax1.legend(loc='best', fontsize=9)
        ax1.xaxis.set_major_formatter(date_fmt)
        
        # Plot 2: Momentum Indicators
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['RSI'], label='RSI', color='#d62728', linewidth=1.5)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.fill_between(self.data.index, 30, 70, alpha=0.1, color='gray')
        
        # MACD
        ax2_macd = ax2.twinx()
        ax2_macd.plot(self.data.index, self.data['MACD'], label='MACD', color='#17becf', linewidth=1.5, alpha=0.7)
        
        ax2.set_title('Momentum Indicators', fontsize=14)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left', fontsize=9)
        ax2_macd.legend(loc='upper right', fontsize=9)
        ax2.xaxis.set_major_formatter(date_fmt)
        
        # Plot 3: Volume and ADX
        ax3 = axes[2]
        # Volume bars
        ax3.bar(self.data.index, self.data['Volume'], color='#aec7e8', alpha=0.6, label='Volume')
        
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
            'Price_SMA200_Ratio': [self.latest['Close']/self.latest['SMA200']],
            'Entry_Targets': ' - '.join(f"{x:.2f}" for x in self.price_targets['entry']),
            'Exit_Targets': ' - '.join(f"{x:.2f}" for x in self.price_targets['exit']),
            'Signals': ' | '.join(self.signals)
        }
        
        # Add fundamental data if available
        for key, value in self.fundamental_data.items():
            if value is not None:
                summary_data[key] = [value]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filename, index=False)
        print(f"💾 Report saved as: {filename}")
    
def get_stock_recommendations():
    """Recommend stocks based on strong technical and fundamental patterns"""
    print("\n🔍 Generating stock recommendations...")
    
    # In a real implementation, this would use screening APIs
    # For demo purposes, we'll return a static list
    recommendations = [
        {"ticker": "MSFT", "reason": "Strong technicals with Golden Cross pattern, positive EPS growth"},
        {"ticker": "JNJ", "reason": "Stable dividend payer with defensive characteristics"},
        {"ticker": "HD", "reason": "Strong technical momentum and institutional support"},
        {"ticker": "V", "reason": "Consistent performer with strong fundamentals"},
        {"ticker": "DIS", "reason": "Undervalued with positive sentiment turnaround"}
    ]
    
    print("\n🌟 TOP LONG-TERM STOCK RECOMMENDATIONS:")
    print("=" * 70)
    for i, stock in enumerate(recommendations, 1):
        print(f"{i}. {stock['ticker']}: {stock['reason']}")
    
    return [stock['ticker'] for stock in recommendations]

def main():
    """Main function with enhanced user interface"""
    print("\n" + "=" * 70)
    print("🌟 ENHANCED STOCK ANALYSIS TOOL FOR LONG-TERM INVESTORS")
    print("=" * 70)
    print("📈 Combines Technical Analysis, Fundamentals, and News Sentiment")
    print("💼 Optimized for 6+ Month Investment Horizons")
    print("\n" + "-" * 70)
    
    # Check if user wants recommendations
    want_recommendations = input("\nWould you like stock recommendations? (y/n): ").strip().lower() == 'y'
    if want_recommendations:
        recommended_tickers = get_stock_recommendations()
        print("\n💡 Tip: Analyze these stocks using the tool below")
    
    # Analysis configuration
    period_map = {'1': '6mo', '2': '1y', '3': '2y', '4': '5y', '5': 'max'}
    print("\n📅 Select time period for analysis:")
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
    analyzed_tickers = []
    while True:
        print("\n" + "=" * 70)
        ticker = input("\n📊 Enter stock ticker (or 'done' to finish): ").strip().upper()
        
        if ticker == 'DONE':
            break
        if not ticker or len(ticker) > 5:
            print("❌ Invalid ticker symbol")
            continue
        if ticker in analyzed_tickers:
            print("ℹ️ Already analyzed this stock")
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
        
        analyzed_tickers.append(ticker)
    
    # Final output and suggestions
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("💡 Remember: Technical analysis is one tool - always consider fundamentals and diversification")
    
    print("\n🔮 SUGGESTIONS FOR FURTHER IMPROVEMENT:")
    print("-" * 70)
    print("1. Portfolio Integration:")
    print("   • Add portfolio tracking with allocation analysis")
    print("   • Implement sector diversification scoring")
    print("   • Add risk-adjusted return metrics (Sharpe ratio)")
    
    print("\n2. Enhanced Fundamental Analysis:")
    print("   • Add cash flow analysis and debt metrics")
    print("   • Incorporate historical valuation ranges")
    print("   • Add competitive analysis metrics")
    
    print("\n3. Advanced Alert System:")
    print("   • Price target alerts via email/SMS")
    print("   • Technical pattern recognition alerts")
    print("   • Earnings report reminders")
    
    print("\n4. User Experience Upgrades:")
    print("   • Interactive web-based dashboard")
    print("   • Mobile app with push notifications")
    print("   • PDF report generation")
    
    print("\n5. Machine Learning Features:")
    print("   • Predictive price modeling")
    print("   • Anomaly detection for unusual activity")
    print("   • Sentiment analysis of earnings calls")
    
    print("\n" + "=" * 70)
    print("📈 Happy Long-Term Investing! 💼💰")


if __name__ == "__main__":
    main()