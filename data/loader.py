# data/loader.py
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from connection.mt5_client import MT5Client
from config import settings

class DataLoader:
    """Több időkeretes adatbetöltés optimalizálva GPU memóriakezeléshez"""
    
    def __init__(self):
        self.mt5 = MT5Client()
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_dataset(
        self,
        symbol: str,
        timeframes: List[str],
        start_date: datetime,
        end_date: datetime,
        n_bars: int = 20000,
        max_retries: int = 3
    ) -> Dict[str, pd.DataFrame]:
        """Több időkeret adatainak párhuzamos betöltése"""
        data = {}
        for timeframe in timeframes:
            for attempt in range(max_retries):
                try:
                    df = self._fetch_timeframe_data(
                        symbol, timeframe, start_date, end_date, n_bars
                    )
                    data[timeframe] = self._post_fetch_processing(df)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        self.logger.error(f"{timeframe} adatbetöltés sikertelen: {str(e)}")
                    continue
        return data

    def _fetch_timeframe_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        n_bars: int
    ) -> pd.DataFrame:
        """MT5-ből történő adatlekérés memóriaoptimalizált módon"""
        if not self.mt5.initialize():
            raise ConnectionError("MT5 kapcsolat sikertelen")

        try:
            rates = self.mt5.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                count=n_bars
            )
            
            if not rates:
                raise ValueError("Üres adatválasz az MT5-től")
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df
            
        finally:
            self.mt5.shutdown()

    def _post_fetch_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Előkészítés a feldolgozás előtt"""
        df = df[~df.index.duplicated(keep='first')]
        df = df.resample('1T').ffill()  # Hiányzó percek kitöltése
        return df[['open', 'high', 'low', 'close', 'tick_volume']]