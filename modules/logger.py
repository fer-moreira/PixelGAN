import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - (%(name)s) - [%(levelname)-8s] - %(message)s',
    datefmt='%H:%M:%S'
)

logging.getLogger("urllib3").level = logging.WARNING
logging.getLogger("PIL").level = logging.WARNING
logging.getLogger("Pillow").level = logging.WARNING