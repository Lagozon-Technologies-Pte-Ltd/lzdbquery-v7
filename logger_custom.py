import logging
from logging.config import dictConfig
logging.basicConfig(level=logging.INFO)

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "json": {"format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'}
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default"},
        "file": {"class": "logging.FileHandler", "filename": "app.log", "formatter": "json"}
    },
    "loggers": {
        "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "app": {"handlers": ["console", "file"], "level": "DEBUG", "propagate": False}
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO"
    }
}
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app")