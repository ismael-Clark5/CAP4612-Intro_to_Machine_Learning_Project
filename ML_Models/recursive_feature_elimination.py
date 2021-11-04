from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
import numpy as np
import data as d
import json


X, y = d.get_training_split()

# Create the RFE object and compute a cross-validated score.
log_regression = LR()

selected_features = dict()
# The "accuracy" scoring shows the proportion of correct classifications

for min_features_to_select in np.arange(20, 120, 20):
    rfe = RFE(estimator=log_regression, step=0.1,
                n_features_to_select=min_features_to_select, verbose=0)

    rfe.fit(X, y)
    features = X.columns[rfe.get_support()]
    selected_features[min_features_to_select.item()] = list(features)

json.dump(selected_features, open('../Datasets/RFE.json', 'w'))


