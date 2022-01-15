from ._version import get_versions
from .pareto import gpu_pareto_front

__version__ = get_versions()["version"]
del get_versions
