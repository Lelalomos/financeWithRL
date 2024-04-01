import logging as logger
import sys

def return_logs(logs_path):
    # logging
    logger.getLogger().setLevel(logger.INFO)
    process_handler = logger.StreamHandler(sys.stdout)
    process_handler = logger.FileHandler(logs_path)
    formatter = logger.Formatter('%(asctime)s - %(levelname)s - %(message)s',datefmt='%d-%b-%y %H:%M:%S')
    process_handler.setFormatter(formatter)
    logger.getLogger().addHandler(process_handler)
    return logger

