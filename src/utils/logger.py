import logging

# Create logger
logger = logging.getLogger('trading_bot')
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create file handler
file_handler = logging.FileHandler('trading_bot.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler) 