from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import numpy as np
import data as d

#Selecting the Best important features according to Logistic Regression using SelectFromModel
X, y = d.get_training_split()

for maximum_features_to_select in np.arange(20,100,20):
    sfm_selector = SelectFromModel(estimator=LogisticRegression())

    sfm_selector.fit(X, y)

    X.columns[sfm_selector.get_support()]
    print(X.columns)