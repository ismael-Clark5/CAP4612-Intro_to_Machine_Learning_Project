from typing import Any
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
import numpy as np
import data as d
import os

## Function used to find the optimal alpha value for LASSO for the given training set.
"""
def findOptimalAlpha() -> Any:
    grid=dict()
    grid['alpha'] = np.arange(0,1,0.01)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    labels = d.training_df['cancer']
    clean_df=d.training_df.drop(labels=['Ensembl_ID', 'cancer', 'Unnamed: 0'], axis=1)
    lasso = LassoCV(alphas=np.arange(0,1,0.01), cv=cv, n_jobs=8)
    results2 = lasso.fit(clean_df, labels)
    print('Alpha: %.3f' % results2.alpha_)
    ##Store our computed alpha value, we dont wanna do extra work if not needed
    f = open('alpha.csv', mode='w')
    f.write(f'{results2.alpha_}')
    return results2
"""

def fitModel(model) -> Any:
    labels = d.training_df['cancer']
    clean_df=d.training_df.drop(labels=['Ensembl_ID', 'cancer', 'Unnamed: 0'], axis=1)
    return model.fit(clean_df, labels)
"""
def predict(model, row, labels):
    prediction = model.predict(row)
    total_count = len(prediction)
    total_correct = 0
    total_wrong = 0
    for x in range(len(prediction)):
        prediction[x] = np.abs(np.round(prediction[x]))
        if prediction[x] == labels[x]:
            total_correct += 1
        else:
            total_wrong  += 1
    print(f'Predicted Class: {prediction}')
    print(f'Total Predictions: {total_count}, Total Correct: {total_correct} Correct %: { (total_correct / total_count) * 100}')

"""

def init():
    alpha=0.0
    model = Lasso()
    pipeline = Pipeline([
        ('model', model)
    ])
    search = GridSearchCV(pipeline,
                          {'model__alpha': np.arange(0.1, 10, 0.1)},
                          cv=5, scoring="neg_mean_squared_error", verbose=3
                          )
    print(search.best_params_)
    search.fit(d.get_training_split())
    print(search.best_params_)
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    print(np.array(importance))
    """
    if(os.path.exists('alpha.csv')):
        print('Found alpha file, loading alpha value from file')
        alpha = float(open('alpha.csv').read())
        print(f'Fitting Model with red alpha of {alpha}')
        model = fitModel(model=Lasso(alpha=alpha))
    else:
        print('Alpha File not Found, Computing Alpha value from training data. (This will take a long time)')
        model = findOptimalAlpha()
        alpha = model.alpha_
    labels = d.testing_df['cancer']
    print(f'Labels len: {len(labels)}')
    clean_df = d.testing_df.drop(labels=['Ensembl_ID', 'cancer', 'Unnamed: 0'], axis=1)
    predict(model, clean_df, labels)

    """

if __name__ == '__main__':
    init()