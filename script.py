"""
Stock Analysis Tool for Long-Term Investors
===================================================
A comprehensive technical analysis tool using free data sources
optimized for swing trading and long-term investment decisions.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

class StockAnalyzer:
    def __init__(self, ticker: str, period: str = "1y", investment_horizon: str = "long-term"):
        self.ticker = ticker.upper()
        self.period = period
        self.investment_horizon = investment_horizon
        self.data = None
        self.latest = None
        self.score = 0
        self.signals = []
        self.confidence_level = "Low"
    
    def fetch_data(self) -> bool:
        try:
            ticker_obj = yf.Ticker(self.ticker)
            self.data = ticker_obj.history(period=self.period)

            if self.data.empty:
                print(f"No data found for ticker: {self.ticker}")
                return False

            try:
                info = ticker_obj.info
                self.company_name = info.get('longName', self.ticker)
                self.sector = info.get('sector', 'Unknown')
                self.market_cap = info.get('marketCap', 'Unknown')
            except:
                self.company_name = self.ticker
                self.secotr = 'Unknown'
                self.market_cap = 'Unknown'
            
            self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
            print(f"Successfully loaded {len(self.data)} days of data for {self.ticker}")
            return True
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return False
    
    def calculate_indicators(self):
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        volume = self.data['Volume']

        #RSI
        delta = close.diff()
        gain = delta.clip(lower = 0)
        loss = -delta.clip(upper = 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain/avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # 2. Stochastic RSI (more sensitive for long-term)
        rsi_min = self.data['RSI'].rolling(window=14).min()
        rsi_max = self.data['RSI'].rolling(window=14).max()
        self.data['StochRSI'] = (self.data['RSI'] - rsi_min) / (rsi_max - rsi_min) * 100
        
        # 3. Multiple Moving Averages
        self.data['SMA20'] = close.rolling(window=20).mean()
        self.data['SMA50'] = close.rolling(window=50).mean()
        self.data['SMA200'] = close.rolling(window=200).mean()
        self.data['EMA20'] = close.ewm(span=20).mean()
        self.data['EMA50'] = close.ewm(span=50).mean()
        
        # 4. MACD with improved parameters
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        self.data['MACD'] = ema12 - ema26
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']
        
        # 5. Bollinger Bands
        self.data['BB_Mid'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        self.data['BB_Upper'] = self.data['BB_Mid'] + 2 * bb_std
        self.data['BB_Lower'] = self.data['BB_Mid'] - 2 * bb_std
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Mid']
        
        # 6. Average True Range (ATR) for volatility
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['ATR'] = true_range.rolling(window=14).mean()
        
        # 7. ADX (Average Directional Index) for trend strength
        self.data['ADX'] = self._calculate_adx()
        
        # 8. Volume indicators
        self.data['Volume_SMA20'] = volume.rolling(window=20).mean()
        self.data['Volume_Ratio'] = volume / self.data['Volume_SMA20']
        
        # 9. Price momentum
        self.data['Momentum_10'] = close / close.shift(10) - 1
        self.data['Momentum_20'] = close / close.shift(20) - 1
        
        # 10. Support/Resistance levels
        self.data['Resistance'] = high.rolling(window=20).max()
        self.data['Support'] = low.rolling(window=20).min()

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

        self._calculate_confidence()
    
    def _analyze_trend(self) -> float:
        score = 0
        close = self.latest['Close']

        # Moving Average Alignment
        if close > self.latest['SMA20'] > self.latest['SMA50'] > self.latest['SMA200']:
            score += 3
            self.signals.append("🟢 Strong uptrend: Price above all major moving averages")
        elif close > self.latest['SMA50'] > self.latest['SMA200']:
            score += 2
            self.signals.append("🟢 Moderate uptrend: Price above medium and long-term averages")
        elif close < self.latest['SMA20'] < self.latest['SMA50'] < self.latest['SMA200']:
            score -= 3
            self.signals.append("🔴 Strong downtrend: Price below all major moving averages")
        elif close < self.latest['SMA50'] < self.latest['SMA200']:
            score -= 2
            self.signals.append("🔴 Moderate downtrend: Price below medium and long-term averages")
        
        # ADX Trend Strength
        if self.latest['ADX'] > 25:
            if close > self.latest['SMA50']:
                score += 1
                self.signals.append("💪 Strong upward trend confirmed by ADX")
            else:
                score -= 1
                self.signals.append("⚠️ Strong downward trend confirmed by ADX")
        elif self.latest['ADX'] < 20:
            self.signals.append("😴 Weak trend - market is consolidating")
        
        return score
    
    def _analyze_momentum(self) -> float:
        score = 0

        # RSI Analysis
        rsi = self.latest['RSI']
        rsi_prev = self.prev['RSI']
        
        if rsi < 30 and rsi > rsi_prev:
            score += 2
            self.signals.append("📈 RSI oversold and turning upward - potential reversal")
        elif rsi > 70 and rsi < rsi_prev:
            score -= 2
            self.signals.append("📉 RSI overbought and turning downward - potential reversal")
        elif 40 < rsi < 60:
            self.signals.append("⚖️ RSI neutral - no clear momentum signal")
        
        # Stochastic RSI
        stoch_rsi = self.latest['StochRSI']
        if stoch_rsi < 20:
            score += 1
            self.signals.append("🔄 Stochastic RSI oversold - watch for reversal")
        elif stoch_rsi > 80:
            score -= 1
            self.signals.append("🔄 Stochastic RSI overbought - watch for reversal")
        
        # MACD Analysis
        macd = self.latest['MACD']
        macd_signal = self.latest['MACD_Signal']
        macd_hist = self.latest['MACD_Histogram']
        macd_hist_prev = self.prev['MACD_Histogram']
        
        if macd > macd_signal and macd_hist > macd_hist_prev:
            score += 1.5
            self.signals.append("🚀 MACD bullish crossover with increasing momentum")
        elif macd < macd_signal and macd_hist < macd_hist_prev:
            score -= 1.5
            self.signals.append("📉 MACD bearish crossover with decreasing momentum")
        
        # Price Momentum
        mom_20 = self.latest['Momentum_20']
        if mom_20 > 0.05:  # 5% gain over 20 days
            score += 1
            self.signals.append("⚡ Strong 20-day price momentum")
        elif mom_20 < -0.05:  # 5% loss over 20 days
            score -= 1
            self.signals.append("⚡ Weak 20-day price momentum")
        
        return score

    def _analyze_mean_reversion(self) -> float:
        """Analyze mean reversion opportunities"""
        score = 0
        close = self.latest['Close']
        
        # Bollinger Bands
        bb_position = (close - self.latest['BB_Lower']) / (self.latest['BB_Upper'] - self.latest['BB_Lower'])
        
        if bb_position < 0.1:  # Near lower band
            score += 1.5
            self.signals.append("🎯 Price near lower Bollinger Band - potential bounce")
        elif bb_position > 0.9:  # Near upper band
            score -= 1.5
            self.signals.append("⚠️ Price near upper Bollinger Band - potential pullback")
        
        # Bollinger Band squeeze (low volatility)
        if self.latest['BB_Width'] < np.percentile(self.data['BB_Width'].dropna(), 20):
            self.signals.append("🤐 Bollinger Band squeeze - potential breakout coming")
        
        return score

    def _analyze_volume(self) -> float:
        """Analyze volume patterns"""
        score = 0
        vol_ratio = self.latest['Volume_Ratio']
        
        if vol_ratio > 1.5:
            score += 0.5
            self.signals.append("📊 High volume confirms price movement")
        elif vol_ratio < 0.5:
            score -= 0.5
            self.signals.append("📊 Low volume - lack of conviction")
        
        return score
    
    def _analyze_volatility(self) -> float:
        """Analyze volatility for risk assessment"""
        score = 0
        current_atr = self.latest['ATR']
        avg_atr = self.data['ATR'].tail(50).mean()
        
        if current_atr > avg_atr * 1.3:
            self.signals.append("⚡ High volatility - increased risk and opportunity")
        elif current_atr < avg_atr * 0.7:
            self.signals.append("😴 Low volatility - stable but limited upside")
        
        return score

    def _calculate_confidence(self):
        """Calculate confidence level based on signal strength"""
        signal_count = len([s for s in self.signals if '🟢' in s or '🔴' in s])
        
        if abs(self.score) >= 6 and signal_count >= 4:
            self.confidence_level = "High"
        elif abs(self.score) >= 3 and signal_count >= 2:
            self.confidence_level = "Medium"
        else:
            self.confidence_level = "Low"
        
    
    def get_recommendation(self) -> str:
        """Generate final recommendation"""
        if self.score >= 6:
            return "STRONG BUY"
        elif self.score >= 3:
            return "BUY"
        elif self.score <= -6:
            return "STRONG SELL"
        elif self.score <= -3:
            return "SELL"
        else:
            return "HOLD"

    def get_recommendation_emoji(self) -> str:
        """Get emoji for recommendation"""
        rec = self.get_recommendation()
        emoji_map = {
            "STRONG BUY": "🚀",
            "BUY": "🟢",
            "HOLD": "🟡",
            "SELL": "🔴",
            "STRONG SELL": "💀"
        }
        return emoji_map.get(rec, "🟡")

    def print_analysis(self):
        """Print comprehensive analysis results"""
        rec = self.get_recommendation()
        emoji = self.get_recommendation_emoji()
        
        print("=" * 80)
        print(f"📊 STOCK ANALYSIS REPORT")
        print("=" * 80)
        print(f"Company: {self.company_name}")
        print(f"Ticker: {self.ticker} | Sector: {self.sector}")
        print(f"Analysis Period: {self.period.upper()} | Investment Horizon: {self.investment_horizon.title()}")
        print(f"Current Price: ${self.latest['Close']:.2f}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print()
        
        print(f"🎯 RECOMMENDATION: {emoji} {rec}")
        print(f"📈 Signal Score: {self.score}/10")
        print(f"🎯 Confidence Level: {self.confidence_level}")
        print()
        
        print("📋 DETAILED SIGNALS:")
        print("-" * 40)
        if self.signals:
            for i, signal in enumerate(self.signals, 1):
                print(f"{i:2d}. {signal}")
        else:
            print("No clear signals detected.")
        
        print()
        print("💡 INVESTMENT GUIDANCE:")
        print("-" * 40)
        self._print_investment_guidance(rec)
        
        print()
        print("📊 KEY METRICS:")
        print("-" * 40)
        self._print_key_metrics()
    
    def _print_investment_guidance(self, recommendation: str):
        """Print investment guidance based on recommendation"""
        guidance = {
            "STRONG BUY": [
                "Multiple strong bullish signals detected",
                "Consider initiating or adding to position",
                "Set stop-loss 10-15% below current price",
                "Monitor for any trend reversal signals"
            ],
            "BUY": [
                "Positive momentum with good entry opportunity",
                "Consider dollar-cost averaging into position",
                "Watch for volume confirmation on breakouts",
                "Set stop-loss 8-12% below current price"
            ],
            "HOLD": [
                "Mixed signals suggest waiting for clearer direction",
                "If already holding, maintain position",
                "Look for breakout above resistance or support test",
                "Monitor upcoming earnings or news catalysts"
            ],
            "SELL": [
                "Weakening momentum suggests profit-taking opportunity",
                "Consider reducing position size if profitable",
                "Watch for any reversal signals before full exit",
                "Implement trailing stop-loss strategy"
            ],
            "STRONG SELL": [
                "Multiple bearish signals suggest downtrend continuation",
                "Consider exiting position to preserve capital",
                "If shorting, use tight stop-loss above resistance",
                "Wait for clear reversal before re-entry"
            ]
        }
        
        for point in guidance.get(recommendation, []):
            print(f"• {point}")
        
    def _print_key_metrics(self):
        """Print key technical metrics"""
        print(f"RSI (14): {self.latest['RSI']:.1f}")
        print(f"MACD: {self.latest['MACD']:.3f}")
        print(f"ADX: {self.latest['ADX']:.1f}")
        print(f"20-day Momentum: {self.latest['Momentum_20']*100:.1f}%")
        print(f"Volume Ratio: {self.latest['Volume_Ratio']:.1f}x")
        print(f"Distance from SMA50: {((self.latest['Close']/self.latest['SMA50'])-1)*100:.1f}%")

    def plot_analysis(self, save_plot: bool = False):
        """Create comprehensive analysis plots"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        fig.suptitle(f'{self.ticker} - Technical Analysis ({self.period.upper()})', fontsize=16, fontweight='bold')
        
        # Plot 1: Price and Moving Averages
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2, color='black')
        ax1.plot(self.data.index, self.data['SMA20'], label='SMA 20', alpha=0.7, color='blue')
        ax1.plot(self.data.index, self.data['SMA50'], label='SMA 50', alpha=0.7, color='orange')
        ax1.plot(self.data.index, self.data['SMA200'], label='SMA 200', alpha=0.7, color='red')
        
        # Bollinger Bands
        ax1.fill_between(self.data.index, self.data['BB_Upper'], self.data['BB_Lower'], 
                        alpha=0.1, color='gray', label='Bollinger Bands')
        ax1.plot(self.data.index, self.data['BB_Upper'], alpha=0.5, color='gray', linestyle='--')
        ax1.plot(self.data.index, self.data['BB_Lower'], alpha=0.5, color='gray', linestyle='--')
        
        ax1.set_title('Price Action with Moving Averages & Bollinger Bands')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RSI and Stochastic RSI
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['RSI'], label='RSI', color='purple', linewidth=2)
        ax2.plot(self.data.index, self.data['StochRSI'], label='Stochastic RSI', color='orange', alpha=0.7)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        ax2.fill_between(self.data.index, 30, 70, alpha=0.1, color='yellow')
        ax2.set_title('RSI & Stochastic RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: MACD
        ax3 = axes[2]
        ax3.plot(self.data.index, self.data['MACD'], label='MACD', color='blue', linewidth=2)
        ax3.plot(self.data.index, self.data['MACD_Signal'], label='Signal Line', color='red', linewidth=2)
        ax3.bar(self.data.index, self.data['MACD_Histogram'], label='Histogram', alpha=0.3, color='green')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('MACD Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Volume
        ax4 = axes[3]
        ax4.bar(self.data.index, self.data['Volume'], alpha=0.6, color='lightblue', label='Volume')
        ax4.plot(self.data.index, self.data['Volume_SMA20'], color='red', linewidth=2, label='20-day Avg Volume')
        ax4.set_title('Volume Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"{self.ticker}_analysis_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📁 Chart saved as: {filename}")
        
        plt.show()

    def save_report(self, filename: Optional[str] = None):
        """Save analysis report to CSV"""
        if filename is None:
            filename = f"{self.ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        # Create summary data
        summary_data = {
            'Ticker': [self.ticker],
            'Company': [self.company_name],
            'Analysis_Date': [datetime.now().strftime('%Y-%m-%d %H:%M')],
            'Recommendation': [self.get_recommendation()],
            'Score': [self.score],
            'Confidence': [self.confidence_level],
            'Current_Price': [self.latest['Close']],
            'RSI': [self.latest['RSI']],
            'MACD': [self.latest['MACD']],
            'ADX': [self.latest['ADX']],
            'Volume_Ratio': [self.latest['Volume_Ratio']],
            'Signals': [' | '.join(self.signals)]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filename, index=False)
        print(f"📁 Report saved as: {filename}")
    
def main():
    """Main function with user interface"""
    print("🏗️ Enhanced Stock Analysis Tool")
    print("=" * 50)
    print("📈 Optimized for Long-Term & Swing Trading")
    print()
    
    # Get user inputs
    while True:
        ticker = input("📊 Enter stock ticker (e.g., AAPL, MSFT, TSLA): ").strip().upper()
        if ticker and len(ticker) <= 10:
            break
        print("❌ Please enter a valid ticker symbol")
    
    print("\n📅 Select time period:")
    print("1. 6 months (6mo)")
    print("2. 1 year (1y) - Recommended")
    print("3. 2 years (2y)")
    print("4. 5 years (5y)")
    
    period_map = {'1': '6mo', '2': '1y', '3': '2y', '4': '5y'}
    period_choice = input("Choose period (1-4) or enter custom (e.g., '1y'): ").strip()
    period = period_map.get(period_choice, period_choice if period_choice else '1y')
    
    print("\n🎯 Select investment horizon:")
    print("1. Swing Trading (days to weeks)")
    print("2. Long-term Investment (months to years)")
    
    horizon_choice = input("Choose horizon (1-2): ").strip()
    horizon = "swing-trading" if horizon_choice == "1" else "long-term"
    
    print("\n📊 Analysis Options:")
    show_plot = input("Show technical charts? (y/n): ").strip().lower() == 'y'
    save_plot = input("Save chart as image? (y/n): ").strip().lower() == 'y' if show_plot else False
    save_report = input("Save analysis report? (y/n): ").strip().lower() == 'y'
    
    print("\n🔍 Analyzing... Please wait...")
    
    # Create analyzer and run analysis
    analyzer = StockAnalyzer(ticker, period, horizon)
    
    if not analyzer.fetch_data():
        return
    
    analyzer.calculate_indicators()
    analyzer.analyze_signals()
    analyzer.print_analysis()
    
    if show_plot:
        analyzer.plot_analysis(save_plot=save_plot)
    
    if save_report:
        analyzer.save_report()
    
    print("\n✅ Analysis complete!")
    print("💡 Tip: Always do your own research and consider multiple factors before investing!")


if __name__ == "__main__":
    main()