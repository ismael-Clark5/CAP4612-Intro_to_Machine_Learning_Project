from typing import Any
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
import numpy as np
import data as d
import os


def init():
    alpha=0.0
    model = Lasso()
    pipeline = Pipeline([
        ('model', model)
    ])
    search = GridSearchCV(pipeline,
                          {'model__alpha': np.arange(0.1, 10, 0.1)},
                          cv=5, scoring="neg_mean_squared_error", verbose=1
    )
    X, Y = d.get_training_split()
    search.fit(X, Y)
    print(search.best_params_)
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    print(f'Importance: {len(importance)}, Values: {importance}')
    test = X.columns.to_numpy()[importance > 0]
    print(test)

if __name__ == '__main__':
    init()