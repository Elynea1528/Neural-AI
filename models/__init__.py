# models/__init__.py (javított verzió)
"""Modellek fő inicializációs fájlja"""
from .common.base import BaseLightningModel
from .utils.factory import ModelFactory

# Architektúrák közvetlen importálása
from .architectures import (
    GRULightning,
    ICMLightning,
    NBEATSLightning,
    MLPLightning
)

__all__ = [
    'ModelFactory',
    'BaseLightningModel',
    'GRULightning',
    'ICMLightning',
    'NBEATSLightning',
    'MLPLightning'
]