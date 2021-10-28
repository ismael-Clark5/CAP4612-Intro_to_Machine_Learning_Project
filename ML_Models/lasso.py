from typing import Any
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
import numpy as np
import data as d
import json


def init():
    features_found=dict()
    for alpha in np.arange(0.001,1,0.05):
        model = Lasso(alpha=alpha)
        X, Y = d.get_training_split()
        model.fit(X, Y)
        coefficients = model.coef_
        importance = np.abs(coefficients)
        features = X.columns.to_numpy()[importance > 0]
        features_found[alpha] = list(features)
    json.dump(features_found, open('../Datasets/LASSO.json', 'w'))
if __name__ == '__main__':
    init()