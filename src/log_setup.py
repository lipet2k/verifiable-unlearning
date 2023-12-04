from loguru import logger as log
import sys

log.remove(0)
log.add(sys.stdout, level="TRACE")