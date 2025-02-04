# data/__init__.py
"""Adatkezelő modulok összekapcsolása"""
from .loader import DataLoader
from .processor import DataProcessor
from .saver import AdvancedDataSaver

__all__ = ['DataLoader', 'DataProcessor', 'AdvancedDataSaver']