from .calc import Calculator
from pkg_resources import get_distribution, DistributionNotFound

version = get_distribution(__name__).version
