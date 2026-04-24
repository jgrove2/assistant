import os
import logging

logger = logging.getLogger(__name__)


def get_int_env(name: str, default: int) -> int:
    try:
        value = os.environ.get(name)
        return int(value) if value is not None else default
    except ValueError:
        logger.warning("Invalid integer value for %s, using default %d", name, default)
        return default


def get_float_env(name: str, default: float) -> float:
    try:
        value = os.environ.get(name)
        return float(value) if value is not None else default
    except ValueError:
        logger.warning("Invalid float value for %s, using default %f", name, default)
        return default
