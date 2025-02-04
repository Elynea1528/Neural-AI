import os
from pathlib import Path
from typing import Dict, Any

class Settings:
    """Alap konfigurációs beállítások"""
    
    # Projekt struktúra
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    MODELS_DIR = PROJECT_ROOT / "trained_models"
    
    # Modell paraméterek
    MODEL_CONFIG: Dict[str, Any] = {
        'input_size': 20,
        'hidden_size': 128,
        'num_layers': 2,
        'learning_rate': 1e-3,
        'dropout': 0.2
    }
    
    # Adatfeldolgozás
    DATA_CONFIG: Dict[str, Any] = {
        'window_size': 30,
        'train_ratio': 0.8,
        'batch_size': 64,
        'num_workers': os.cpu_count() or 4
    }
    
    # Logging konfiguráció
    LOGGING_CONFIG: Dict[str, Any] = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': LOGS_DIR / "app.log"
    }
    
# Singleton példány
settings = Settings()