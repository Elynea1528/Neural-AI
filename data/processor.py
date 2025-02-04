# data/processor.py
import pandas as pd
import numpy as np
import pandas_ta as ta
import torch
from typing import Dict, Tuple
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

class FeatureEngineer:
    """Haladó jellemzőmérnöki műveletek LSTM/Transformer modellekhez"""
    
    @staticmethod
    def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Időalapú jellemzők gazdagítása"""
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        return df

    @staticmethod
    def add_advanced_ta_features(df: pd.DataFrame) -> pd.DataFrame:
        """Korszerű technikai indikátorok hozzáadása"""
        # Momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # Volatilitás
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        df['volatility_30'] = df['close'].pct_change().rolling(30).std()
        
        # Összetett indikátorok
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['MACD'] = ta.macd(df['close']).iloc[:, 0]
        df['Bollinger_%'] = (df['close'] - ta.bbands(df['close']).iloc[:, 2]) / (
            ta.bbands(df['close']).iloc[:, 0] - ta.bbands(df['close']).iloc[:, 2]
        )
        
        # Volume alapú jellemzők
        if 'tick_volume' in df.columns:
            df['volume_zscore'] = (df['tick_volume'] - df['tick_volume'].rolling(50).mean()) / df['tick_volume'].rolling(50).std()
            df['volume_roc'] = df['tick_volume'].pct_change(5)
            
        return df.dropna()

class DataProcessor:
    """Adatfeldolgozási pipeline GPU optimalizált tenzorokkal"""
    
    def __init__(self, seq_length: int = 60, pred_length: int = 3):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.scalers = {}
        self.feature_engineer = FeatureEngineer()

    def full_pipeline(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, torch.Tensor]:
        """Teljes feldolgozási folyamat több időkerethez"""
        processed = {}
        for timeframe, df in raw_data.items():
            try:
                df = self.feature_engineer.add_temporal_features(df)
                df = self.feature_engineer.add_advanced_ta_features(df)
                df = self._normalize_features(df)
                sequences = self._create_sequences(df)
                processed[timeframe] = self._to_gpu_tensors(sequences)
            except Exception as e:
                raise RuntimeError(f"{timeframe} feldolgozási hiba: {str(e)}")
        return processed

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligens normalizáció jellemzőtípus alapján"""
        price_cols = ['open', 'high', 'low', 'close']
        volatility_cols = [c for c in df.columns if 'volatility' in c]
        volume_cols = [c for c in df.columns if 'volume' in c]
        
        # Árak: RobustScaling
        self.scalers['price'] = RobustScaler()
        df[price_cols] = self.scalers['price'].fit_transform(df[price_cols])
        
        # Volatilitás: Log transform + MinMax
        df[volatility_cols] = np.log1p(df[volatility_cols])
        self.scalers['volatility'] = MinMaxScaler()
        df[volatility_cols] = self.scalers['volatility'].fit_transform(df[volatility_cols])
        
        # Volume: Z-score normalization
        if volume_cols:
            self.scalers['volume'] = RobustScaler()
            df[volume_cols] = self.scalers['volume'].fit_transform(df[volume_cols])
            
        return df

    def _create_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """3D szekvenciák létrehozása (samples, seq_len, features)"""
        features = df.values
        num_samples = len(features) - self.seq_length - self.pred_length + 1
        sequences = np.lib.stride_tricks.sliding_window_view(
            features, (self.seq_length + self.pred_length, features.shape[1])
        )
        return sequences.reshape(-1, self.seq_length + self.pred_length, features.shape[1])

    def _to_gpu_tensors(self, data: np.ndarray) -> Dict[str, torch.Tensor]:
        """Adatok átalakítása GPU-optimalizált tenzorokká"""
        return {
            'inputs': torch.tensor(data[:, :self.seq_length, :], 
                                 dtype=torch.float32).pin_memory().cuda(),
            'targets': torch.tensor(data[:, self.seq_length:, 3],  # close ár
                                  dtype=torch.float32).pin_memory().cuda()
        }

    def temporal_split(self, data: Dict[str, torch.Tensor], test_size: float = 0.2) -> Tuple:
        """Időalapú felosztás több időkeretre"""
        split_data = {}
        for timeframe, tensors in data.items():
            split_idx = int(tensors['inputs'].shape[0] * (1 - test_size))
            split_data[timeframe] = {
                'train': (tensors['inputs'][:split_idx], tensors['targets'][:split_idx]),
                'val': (tensors['inputs'][split_idx:], tensors['targets'][split_idx:])
            }
        return split_data