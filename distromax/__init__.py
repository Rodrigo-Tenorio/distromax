from ._version import get_versions
from .logging import _get_default_logger, set_up_logger

__version__ = get_versions()["version"]
del get_versions

try:
    logger = _get_default_logger()
    logger.info(f"Running distromax version {__version__}")
except Exception as e:  # pragma: no cover
    print(
        f"Logging setup failed with exception: {e}\n"
        "Proceeding without default logging."
    )

from .analytical import AnalyticalGammaToGumbel

from .fit import (
    BatchMaxGumbel,
    BatchMaxGumbelNotchingOutliers,
)
