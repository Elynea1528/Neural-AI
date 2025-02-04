# data/saver.py
import joblib
import logging
import numpy as np
import pandas as pd
import torch  # Hiányzó import hozzáadva
from pathlib import Path
from typing import Dict, Any, List
from config import settings


class AdvancedDataSaver:
    """Haladó adatmentés HDF5 és NPZ formátumban egyaránt"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.save_dir = settings.DATA_DIR / 'processed'
        self.save_dir.mkdir(exist_ok=True)
        
    def save_processed_data(
        self,
        data: Dict[str, Any],
        experiment_name: str,
        formats: List[str] = ['hdf', 'npz', 'parquet']
    ) -> None:
        """Adatok mentése több formátumban"""
        try:
            base_path = self.save_dir / experiment_name
            base_path.mkdir(exist_ok=True)
            
            # Scaler mentés
            joblib.dump(data['scalers'], base_path / 'scalers.joblib')
            
            # Tenzor mentés
            for timeframe, tensors in data['tensors'].items():
                time_path = base_path / timeframe
                time_path.mkdir(exist_ok=True)
                
                if 'hdf' in formats:
                    pd.DataFrame(tensors['inputs'].cpu().numpy().reshape(-1, tensors['inputs'].shape[-1])).to_hdf(
                        time_path / 'inputs.h5', key='data', mode='w'
                    )
                
                if 'npz' in formats:
                    np.savez(
                        time_path / 'data.npz',
                        inputs=tensors['inputs'].cpu().numpy(),
                        targets=tensors['targets'].cpu().numpy()
                    )
                    
                if 'parquet' in formats:
                    pd.DataFrame({
                        'inputs': list(tensors['inputs'].cpu().numpy()),
                        'targets': list(tensors['targets'].cpu().numpy())
                    }).to_parquet(time_path / 'data.parquet')
                    
            self.logger.info(f"Adatok elmentve ide: {base_path}")
            
        except Exception as e:
            self.logger.error(f"Mentési hiba: {str(e)}")
            raise

    def load_processed_data(self, experiment_name: str, timeframe: str) -> Dict[str, torch.Tensor]:
        """Adatok betöltése GPU-ra direkt módon"""
        data_path = self.save_dir / experiment_name / timeframe
        return {
            'inputs': torch.load(data_path / 'inputs.pt').cuda(),
            'targets': torch.load(data_path / 'targets.pt').cuda()
        }