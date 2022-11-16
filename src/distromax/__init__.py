from importlib.metadata import version
from .logging import _get_default_logger, set_up_logger

__version__ = version(__name__)

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
