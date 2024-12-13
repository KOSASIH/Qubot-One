# src/utils/logging_utils.py

import logging

class Logger:
    @staticmethod
    def setup_logger(name, log_file, level=logging.INFO):
        """Set up a logger.

        Args:
            name (str): Name of the logger.
            log_file (str): Log file path.
            level (int): Logging level.
        """
        logger = logging.getLogger(name)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        print(f"Logger {name} set up with log file {log_file}.")

    @staticmethod
    def log_info(logger, message):
        """Log an info message.

        Args:
            logger (logging.Logger): Logger instance.
            message (str): Message to log.
        """
        logger.info(message)
        print(f"INFO: {message}")

    @staticmethod
    def log_error(logger, message):
        """Log an error message.

        Args:
            logger (logging.Logger): Logger instance.
            message (str): Message to log.
        """
        logger.error(message)
        print(f"ERROR: {message}")
