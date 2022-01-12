import logging


class ILoggable:
    def __init__(self, logger_name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(logger_name)
        if self.logger.level != level:
            self.logger.setLevel(level)
