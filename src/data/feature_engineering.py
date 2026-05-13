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
        df['BOS_Bullish'] = df['close'] > df['high'].shift(1).rolling(5).max()
        df['CHOCH_Bearish'] = df['close'] < df['low'].shift(1).rolling(5).min()
        return df
