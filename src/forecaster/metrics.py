from typing import Callable, Dict, List
import numpy as np
import properscoring as ps
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

EPS = 1e-6

def rmse(predictions, targets):
    """
    Root Mean Squared Error
    Args:
        predictions (np.ndarray): Point Predictions of the model
        targets (np.ndarray): Point Targets of the model
    Returns:
        float: RMSE
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def norm_rmse(predictions, targets):
    """
    Root Mean Squared Error
    Args:
        predictions (np.ndarray): Point Predictions of the model
        targets (np.ndarray): Point Targets of the model
    Returns:
        float: RMSE
    """
    try:
        scale = MinMaxScaler()
        targets = scale.fit_transform(targets[:,None])
        predictions = scale.transform(predictions[:,None])
    finally:
       return np.sqrt(((predictions - targets) ** 2).mean())


def mape(predictions, targets):
    """
    Mean Absolute Percentage Error
    Args:
        predictions (np.ndarray): Predictions of the model
        targets (np.ndarray): Targets of the model
    Returns:
        float: MAPE
    """
    targets[targets==0] = np.nan
    return np.nanmean(np.abs((predictions - targets) / targets)) * 100
