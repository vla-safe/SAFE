from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class AbstractPredictor(ABC):
    """An abstract class for conformal prediction"""

    @abstractmethod
    def get_prediction_band(
        self, training_data: np.ndarray, calibration_data: np.ndarray, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Obtains upper and lower prediction trajectory, which together constitutes a prediction
        band.

        Args:
            training_data: An np.ndarray object of size (train_size, length).
            calibration_data: An np.ndarray object of size (calibration_size, length).
            alpha: The significance level. The prediction set is valid with probability 1 - alpha.

        Returns:
            A tuple of (uppder_trajectory, lower_trajectory) that constitutes the prediction band,
            where each object is an np.ndarray with shape (1, length).
        """

    @abstractmethod
    def get_one_sided_prediction_band(
        self,
        training_data: np.ndarray,
        calibration_data: np.ndarray,
        alpha: float,
        lower_bound: bool,
    ) -> np.ndarray:
        """Obtains either upper or lower prediction trajectory, depending on the argument
        lower_bound.

        Args:
            training_data: An np.ndarray object of size (train_size, length).
            calibration_data: An np.ndarray object of size (calibration_size, length).
            alpha: The significance level. The prediction set is valid with probability 1 - alpha.
            lower_bound: If True, lower-bound is returned. Otherwise, upper-bound is returned.

        Returns:
            An np.ndarray of shape (1, length) that represents either the lower-boudning or
            upper-bounding trajectory.
        """
