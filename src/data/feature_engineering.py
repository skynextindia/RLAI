import pandas as pd
import numpy as np

class FeatureEngineer:
    def calculate_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        df['FVG_Bullish'] = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2))
        df['FVG_Bearish'] = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2))
        return df
        
    def calculate_atr(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(period).mean()
        return df

    def detect_choch_bos(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Identify Fractal Highs/Lows (Institutional Pivot Points)
        # A Fractal High is a point higher than 2 candles on each side
        df['Fractal_High'] = df['high'][(df['high'] > df['high'].shift(1)) & 
                                        (df['high'] > df['high'].shift(2)) & 
                                        (df['high'] > df['high'].shift(-1)) & 
                                        (df['high'] > df['high'].shift(-2))]
        
        df['Fractal_Low'] = df['low'][(df['low'] < df['low'].shift(1)) & 
                                       (df['low'] < df['low'].shift(2)) & 
                                       (df['low'] < df['low'].shift(-1)) & 
                                       (df['low'] < df['low'].shift(-2))]

        # Forward fill the last known fractal to compare with current price
        last_high = df['Fractal_High'].ffill()
        last_low = df['Fractal_Low'].ffill()

        # 2. Institutional Break of Structure (BOS)
        # Current Close > Last Fractal High (Bullish) or < Last Fractal Low (Bearish)
        df['BOS_Bullish'] = (df['close'] > last_high.shift(1)) & (df['close'].shift(1) <= last_high.shift(1))
        df['BOS_Bearish'] = (df['close'] < last_low.shift(1)) & (df['close'].shift(1) >= last_low.shift(1))

        # 3. Change of Character (CHOCH)
        # For simplicity in this layer, we mark a break of the opposite fractal as a potential CHOCH
        df['CHOCH_Bullish'] = df['BOS_Bullish'] # Simplified mapping for now
        df['CHOCH_Bearish'] = df['BOS_Bearish']
        
        return df
