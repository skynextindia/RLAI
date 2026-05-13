import pandas as pd
from .feature_engineering import FeatureEngineer

class MTFAggregator:
    def __init__(self, timeframes=['1min', '5min', '15min', '1h', '4h']):
        self.timeframes = timeframes
        self.tick_buffer = []
        self.engineer = FeatureEngineer()
        
    def add_tick(self, tick_data):
        self.tick_buffer.append({
            'timestamp': pd.to_datetime(tick_data['timestamp']),
            'price': tick_data['bid'],
            'volume': tick_data['volume']
        })
        
        # Keep buffer memory efficient (last 10000 ticks approx)
        if len(self.tick_buffer) > 10000:
            self.tick_buffer.pop(0)
            
    def aggregate(self):
        if len(self.tick_buffer) < 2:
            return None
            
        df = pd.DataFrame(self.tick_buffer).set_index('timestamp')
        features = {}
        
        for tf in ['5s', '1min', '5min', '15min', '1h', '4h']:
            ohlc = df['price'].resample(tf).ohlc()
            ohlc['volume'] = df['volume'].resample(tf).sum()
            ohlc.dropna(inplace=True)
            
            if len(ohlc) > 0:  
                try:
                    ohlc = self.engineer.calculate_atr(ohlc)
                    ohlc = self.engineer.detect_choch_bos(ohlc)
                except Exception:
                    pass
                
                # Keep full sequence for AI self-learning, not just last row
                features[tf] = ohlc.copy()
                
        return features
