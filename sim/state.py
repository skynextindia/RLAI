
import pandas as pd
import numpy as np

class MTFStateBuilder:
    """
    Handles high-speed multi-timeframe feature aggregation.
    Pre-calculates OHLC windows and aligns them to the tick-stream.
    """
    def __init__(self, ticks_df: pd.DataFrame, config: dict):
        self.config = config
        self.df = ticks_df.copy()
        
        # Ensure timestamp alignment
        if 'time' in self.df.columns:
            self.df['dt'] = pd.to_datetime(self.df['time'], unit='s')
        else:
            # Fallback if no time column
            self.df['dt'] = pd.date_range(start='2024-01-01', periods=len(self.df), freq='100ms')
        
        self.df.set_index('dt', inplace=True)
        self.windows = config.get('windows', {'m1': 100, 'm5': 100, 'm15': 100, 'h1': 50})
        self.mtf_data = {}
        self._precalculate()

    def _precalculate(self):
        """Build Structural OHLCV for all required timeframes."""
        self.df['mid'] = (self.df['bid'] + self.df['ask']) / 2.0
        
        for tf_key, freq in [('m1', '1min'), ('m5', '5min'), ('m15', '15min'), ('h1', 'h')]:
            resampled = self.df['mid'].resample(freq).ohlc().ffill()
            vol = self.df['volume'].resample(freq).sum().fillna(0)
            mtf = pd.concat([resampled, vol], axis=1)
            
            # Channel 1: Log-Returns
            mtf['returns'] = np.log(mtf['close'] / mtf['close'].shift(1)).fillna(0)
            # Channel 2: Normalized Volume
            mtf['vol_norm'] = np.log1p(mtf['volume']).fillna(0)
            # Channel 3: VWAP-Distance (Normalized)
            cum_vol_price = (mtf['close'] * mtf['volume']).cumsum()
            cum_vol = mtf['volume'].cumsum() + 1e-9
            mtf['vwap'] = cum_vol_price / cum_vol
            # Use rolling z-score for vwap_dist to keep it near N(0,1)
            mtf['vwap_dist'] = (mtf['close'] - mtf['vwap']) / (mtf['close'].rolling(100).std() + 1e-9)
            mtf['vwap_dist'] = mtf['vwap_dist'].clip(-10, 10)
            
            # Channel 4: Momentum (Clipped Log-Slope)
            mtf['momentum'] = np.log(mtf['close'] / mtf['close'].shift(5).fillna(mtf['close'])) * 100
            mtf['momentum'] = mtf['momentum'].clip(-10, 10)
            
            # Store 4-channel feature set (Ensure no NaNs)
            self.mtf_data[tf_key] = mtf[['returns', 'vol_norm', 'vwap_dist', 'momentum']].fillna(0).values
            self.df[f'{tf_key}_idx'] = self.df.index.floor(freq)

        self.tick_to_mtf_map = {}
        for tf_key in self.mtf_data.keys():
            freq = {'m1':'1min', 'm5':'5min', 'm15':'15min', 'h1':'h'}[tf_key]
            unique_times = self.df.index.floor(freq).unique()
            time_to_idx = {t: i for i, t in enumerate(unique_times)}
            self.tick_to_mtf_map[tf_key] = self.df.index.floor(freq).map(time_to_idx).values

    def get_mtf_slice(self, tick_idx: int) -> np.ndarray:
        features = []
        for tf_key, window_size in self.windows.items():
            mtf_idx = self.tick_to_mtf_map[tf_key][tick_idx]
            start = max(0, mtf_idx - window_size + 1)
            chunk = self.mtf_data[tf_key][start : mtf_idx + 1]
            
            if len(chunk) < window_size:
                pad = np.zeros((window_size - len(chunk), 4), dtype=np.float32) # Updated to 4 channels
                chunk = np.vstack([pad, chunk])
            features.append(chunk.flatten())
        return np.concatenate(features).astype(np.float32)

    def get_market_metrics(self, tick_idx: int) -> dict:
        """Calculates speed, imbalance, and volatility metrics."""
        # Lookback 100 ticks for metrics
        start = max(0, tick_idx - 100)
        window = self.df.iloc[start : tick_idx + 1].copy()
        window['price_delta'] = window['mid'].diff().fillna(0)
        
        return {
            'market_speed': len(window) / (window.index[-1] - window.index[0]).total_seconds() if len(window) > 1 else 0,
            'volatility': window['mid'].std() if len(window) > 1 else 0,
            'imbalance': (window['volume'] * np.sign(window['price_delta'])).sum() / window['volume'].sum() if window['volume'].sum() > 0 else 0
        }
