# models/utils/factory.py (javított verzió)
"""Modellgyár dinamikus modellkezeléshez"""
from typing import Union, Type, Dict
from models.common.base import BaseLightningModel  # Abszolút import
from models.architectures import (  # Abszolút import
    GRULightning,
    ICMLightning,
    NBEATSLightning,
    MLPLightning
)

class ModelFactory:
    """Dinamikus modelllétrehozás és kezelés"""
    
    _model_registry: Dict[str, Type[BaseLightningModel]] = {
        'gru': GRULightning,
        'icm': ICMLightning,
        'nbeats': NBEATSLightning,
        'mlp': MLPLightning
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        input_size: int,
        **kwargs
    ) -> Union[BaseLightningModel, None]:
        """
        Modell példányosítása a megadott típus alapján
        
        Args:
            model_type (str): A modell típusa (gru, icm, nbeats, mlp)
            input_size (int): Bemeneti jellemzők száma
            **kwargs: Modellspecifikus hiperparaméterek
            
        Returns:
            BaseModel: Az inicializált modell példány
        """
        model_class = cls._model_registry.get(model_type.lower())
        
        if not model_class:
            valid_types = ", ".join(cls._model_registry.keys())
            raise ValueError(f"Érvénytelen modelltípus: {model_type}. "
                             f"Érvényes típusok: {valid_types}")
                             
        return model_class(input_size=input_size, **kwargs)
    
    @classmethod
    def register_model(
        cls, 
        name: str, 
        model_class: Type[BaseLightningModel]
    ) -> None:
        """
        Új modell regisztrálása a gyárba
        
        Args:
            name (str): A modell egyedi azonosítója
            model_class (Type[BaseModel]): A modell osztály
        """
        if not issubclass(model_class, BaseLightningModel):
            raise TypeError("A modellnek a BaseModel leszármazottja kell legyen")
            
        cls._model_registry[name.lower()] = model_class

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """Elérhető modellek listázása"""
        return {k: v.__name__ for k, v in cls._model_registry.items()}