
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
        """Build OHLCV for all required timeframes."""
        for tf_key, freq in [('m1', '1min'), ('m5', '5min'), ('m15', '15min'), ('h1', 'h')]:
            resampled = self.df['last'].resample(freq).ohlc().ffill()
            vol = self.df['volume'].resample(freq).sum().fillna(0)
            mtf = pd.concat([resampled, vol], axis=1)
            
            # Normalize: Log-Returns for price, Log for volume
            mtf['returns'] = np.log(mtf['close'] / mtf['close'].shift(1)).fillna(0)
            mtf['vol_norm'] = np.log1p(mtf['volume']).fillna(0)
            
            # Keep only the features we need to minimize memory
            self.mtf_data[tf_key] = mtf[['returns', 'vol_norm']].values
            
            # Map every tick to its corresponding MTF index for O(1) lookup
            self.df[f'{tf_key}_idx'] = self.df.index.floor(freq)

        # Create a mapping array from tick_idx to mtf_idx
        self.tick_to_mtf_map = {}
        for tf_key in self.mtf_data.keys():
            freq = {'m1':'1min', 'm5':'5min', 'm15':'15min', 'h1':'h'}[tf_key]
            unique_times = self.df.index.floor(freq).unique()
            time_to_idx = {t: i for i, t in enumerate(unique_times)}
            self.tick_to_mtf_map[tf_key] = self.df.index.floor(freq).map(time_to_idx).values

    def get_mtf_slice(self, tick_idx: int) -> np.ndarray:
        """Returns a concatenated MTF feature vector for the given tick index."""
        features = []
        for tf_key, window_size in self.windows.items():
            mtf_idx = self.tick_to_mtf_map[tf_key][tick_idx]
            
            # Slice window, pad with zeros if not enough history
            start = max(0, mtf_idx - window_size + 1)
            chunk = self.mtf_data[tf_key][start : mtf_idx + 1]
            
            if len(chunk) < window_size:
                pad = np.zeros((window_size - len(chunk), chunk.shape[1]), dtype=np.float32)
                chunk = np.vstack([pad, chunk])
            
            features.append(chunk.flatten())
            
        return np.concatenate(features).astype(np.float32)

    def get_market_metrics(self, tick_idx: int) -> dict:
        """Calculates speed, imbalance, and volatility metrics."""
        # Lookback 100 ticks for metrics
        start = max(0, tick_idx - 100)
        window = self.df.iloc[start : tick_idx + 1]
        
        return {
            'market_speed': len(window) / (window.index[-1] - window.index[0]).total_seconds() if len(window) > 1 else 0,
            'volatility': window['last'].std() if len(window) > 1 else 0,
            'imbalance': (window['volume'] * np.sign(window['price_delta'])).sum() / window['volume'].sum() if window['volume'].sum() > 0 else 0
        }
