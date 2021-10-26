from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import numpy as np
import data as d


X, y = d.get_training_split()

# Create the RFE object and compute a cross-validated score.
log_regression = LR()
# The "accuracy" scoring shows the proportion of correct classifications

for min_features_to_select in np.arange(20, 100, 20):
    rfecv = RFECV(estimator=log_regression, step=0.1, cv=StratifiedKFold(2),
                scoring='accuracy',
                min_features_to_select=min_features_to_select)

    rfecv.fit(X, y)
    #print(f'Trying feature_names_in_: {rfecv.feature_names_in_}')
    print("Optimal number of features : %d" % rfecv.n_features_)
#print(f'Output Features: {rfecv.get_feature_names_out(X.columns)}')
