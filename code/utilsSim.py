from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression


XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_,]
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    n_classes = 2  #binary classification
    n_features = 768  # Number of features in dataset =) 4 last layers concatenated
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((1, n_features))  #Since it's binary classification the shape is different than the number of classes
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,)) #Since it's binary classification the shape is different than the number of classes



