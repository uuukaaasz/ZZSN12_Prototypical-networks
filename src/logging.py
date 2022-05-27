import logging
import sys
from os import mkdir, path, remove


def get_logger(log_dir, file_name):
    logger = logging.getLogger()
    if bool(logger.handlers):
        logger.handlers.clear()
    if not path.exists(log_dir):
        mkdir(log_dir)

    log_file_path = path.join(log_dir, file_name)

    if path.exists(log_file_path):
        remove(log_file_path)

    f = open(log_file_path, "w+")
    f.close()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)

    return logger