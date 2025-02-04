from mt5linux import MetaTrader5
import logging

# Naplózás beállítása
logging.basicConfig(level=logging.INFO)

mt5 = MetaTrader5()

def initialize_mt5():
    """Inicializálja a MetaTrader 5 kapcsolatot."""
    try:
        if not mt5.initialize():
            logging.error("MetaTrader5 inicializálás hiba")
            raise RuntimeError("MetaTrader5 inicializálás hiba")
        logging.info("MetaTrader 5 sikeresen inicializálva")
    except Exception as e:
        logging.exception("Hiba történt az MT5 inicializálásakor: %s", e)
        raise
    return mt5  # Visszatér a MetaTrader 5 objektummal

def shutdown_mt5():
    """Leállítja a MetaTrader 5 kapcsolatot."""
    try:
        mt5.shutdown()
        logging.info("MetaTrader 5 kapcsolat sikeresen leállítva.")
    except Exception as e:
        logging.exception("Hiba történt az MT5 leállításakor: %s", e)