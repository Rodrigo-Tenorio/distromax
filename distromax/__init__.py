from .analytical import AnalyticalGammaToGumbel

from .fit import (
    BatchMaxGumbel,
    BatchMaxGumbelNotchingOutliers,
)

from . import _version

__version__ = _version.get_versions()["version"]
