from enum import Enum
from typing import Tuple

import numpy as np

from .abstract_predictor import AbstractPredictor

class RegressionType(Enum):
    Mean = 1

class ModulationType(Enum):
    Const = 1
    Stdev = 2
    Tfunc = 3
    
    
def regress(training_data: np.ndarray, regression_type: RegressionType) -> np.ndarray:
    """Regresses training data to a single trajectory.

    Args:
        training_data: An np.ndarray object of shape (batch_size, length).
        regression_type: A regression type.

    Returns:
        A np.ndarray object of shape (1, length).

    Raises:
        NotImplementedError if unknown regression_type is given.
    """
    if regression_type == RegressionType.Mean:
        return np.mean(training_data, axis=0, keepdims=True)
    else:
        raise (NotImplementedError)



class FunctionalPredictor(AbstractPredictor):
    """Functional conformal prediction method based on `The Importance of Being a Band:
    Finite-Sample Exact Distribution-Free Prediction Sets for Functional Data`
    (https://arxiv.org/abs/2102.06746).

    Attributes:
        modulation_type: Type of modulation function.
        regression_type: Type of regression function (for point prediction).
    """

    def __init__(
        self, modulation_type: ModulationType, regression_type: RegressionType
    ) -> None:
        """Initializes a FunctionalMethod object.

        Args:
            modulation_type: Type of modulation function.
            regression_type: Type of regression function (for point prediction).
        """
        self.modulation_type = modulation_type
        self.regression_type = regression_type

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

        Raises:
            NotImplementedError if unknown regression_type or modulation_type is given.
        """
        length = training_data.shape[-1]
        assert length == calibration_data.shape[-1]
        assert 0.0 < alpha < 1.0

        prediction_trajectory = regress(training_data, self.regression_type)
        modulation_trajectory = self._get_modulation_trajectory(
            training_data, prediction_trajectory, alpha
        )

        calibration_scores = [
            np.max(
                np.abs(calibration_trajectory - prediction_trajectory)
                / modulation_trajectory
            )
            for calibration_trajectory in calibration_data
        ]
        calibration_size = len(calibration_scores)
        band_width = np.sort(calibration_scores)[
            int(np.ceil((calibration_size + 1) * (1 - alpha))) - 1
        ]

        upper_trajectory = prediction_trajectory + band_width * modulation_trajectory
        lower_trajectory = prediction_trajectory - band_width * modulation_trajectory

        assert upper_trajectory.shape == lower_trajectory.shape == (1, length)
        return (upper_trajectory, lower_trajectory)

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
            An np.ndarray of shape (1, length) that represents either the lower-bounding or
            upper-bounding trajectory.

        Raises:
            NotImplementedError if unknown regression_type or modulation_type is given.
        """
        length = training_data.shape[-1]
        assert length == calibration_data.shape[-1]
        assert 0.0 < alpha < 1.0

        prediction_trajectory = regress(training_data, self.regression_type)
        modulation_trajectory = self._get_modulation_trajectory(
            training_data, prediction_trajectory, alpha
        )

        if lower_bound:
            calibration_scores = [
                np.max(
                    (prediction_trajectory - calibration_trajectory)
                    / modulation_trajectory
                )
                for calibration_trajectory in calibration_data
            ]
        else:
            calibration_scores = [
                np.max(
                    (calibration_trajectory - prediction_trajectory)
                    / modulation_trajectory
                )
                for calibration_trajectory in calibration_data
            ]
        band_width = np.quantile(calibration_scores, 1 - alpha)
        # calibration_size = len(calibration_scores)
        # band_width = np.sort(calibration_scores)[
        #     int(np.ceil((calibration_size + 1) * (1 - alpha))) - 1
        # ]
        if lower_bound:
            bounding_trajectory = (
                prediction_trajectory - band_width * modulation_trajectory
            )
        else:
            bounding_trajectory = (
                prediction_trajectory + band_width * modulation_trajectory
            )
        return bounding_trajectory

    def _get_modulation_trajectory(
        self, training_data: np.ndarray, prediction_trajectory: np.ndarray, alpha: float
    ) -> np.ndarray:
        """Obtains modulation trajectory.

        Args:
            training_data: An np.ndarray object of size (train_size, length).
            prediction_trajectory:A np.ndarray object of shape (1, length).
            alpha: The significance level. The prediction set is valid with probability 1 - alpha.

        Returns:
            A np.ndarray object of shape (1, length).

        Raises:
            NotImplementedError if unknown modulation_type is given.
        """
        eps = 1e-8
        length = training_data.shape[-1]
        if self.modulation_type == ModulationType.Const:
            modulation_trajectory = np.ones((1, length)) / length
        elif self.modulation_type == ModulationType.Stdev:
            modulation_trajectory = (
                np.std(training_data, axis=0, ddof=1, keepdims=True) + eps
            )
        elif self.modulation_type == ModulationType.Tfunc:
            train_size = training_data.shape[0]
            if int(np.ceil((train_size + 1) * (1 - alpha))) > train_size:
                modulation_trajectory = (
                    np.max(
                        np.abs(training_data - prediction_trajectory),
                        axis=0,
                        keepdims=True,
                    )
                    + eps
                )
            else:
                gamma = np.sort(
                    np.max(np.abs(training_data - prediction_trajectory), axis=1)
                )[int(np.ceil((train_size + 1) * (1 - alpha))) - 1]
                modulation_trajectory = (
                    np.max(
                        np.abs(training_data - prediction_trajectory)[
                            np.max(
                                np.abs(training_data - prediction_trajectory), axis=1
                            )
                            <= gamma
                        ],
                        axis=0,
                        keepdims=True,
                    )
                    + eps
                )
        else:
            raise (NotImplementedError)
        return modulation_trajectory
