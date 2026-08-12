from importlib.metadata import PackageNotFoundError, version

from standard_precip.base_sp import BaseStandardIndex
from standard_precip.spei import SPEI
from standard_precip.spi import SPI

try:
    __version__ = version("standard-precip")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = ["SPI", "SPEI", "BaseStandardIndex", "__version__"]
