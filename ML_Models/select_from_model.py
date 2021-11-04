from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import numpy as np
import data as d
import json
#Selecting the Best important features according to Logistic Regression using SelectFromModel
X, y = d.get_training_split()

found_features = dict()

for maximum_features_to_select in np.arange(20,120,20):
    sfm_selector = SelectFromModel(estimator=LogisticRegression(), max_features=maximum_features_to_select)

    sfm_selector.fit(X, y)

    features = X.columns[sfm_selector.get_support()]

    found_features[maximum_features_to_select.item()] = list(features)

json.dump(found_features, open('../Datasets/SFM.json', 'w'))