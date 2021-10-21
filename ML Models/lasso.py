from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
import numpy as np
import data as d

lasso_alphas = np.linspace(0,0.2, 11)


model = Lasso()
grid=dict()
grid['alpha'] = np.arange(0,1,0.01)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
labels = d.training_df['cancer']
ensamble_ids=d.training_df['Ensembl_ID']
unamed = d.training_df['Unnamed: 0']
clean_df=d.training_df.drop(labels=['Ensembl_ID', 'cancer', 'Unnamed: 0'], axis=1)
#print(clean_df)
#print(labels)
lasso = LassoCV(alphas=np.arange(0,1,0.1), cv=cv, n_jobs=8)
#search = GridSearchCV(model, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=4)
#scores = cross_val_score(estimator=model, X=clean_df, y=labels, scoring="neg_mean_absolute_error", cv=cv, n_jobs=4)
#print(scores)
#scores = np.absolute(scores)
#print(scores)
#print( 'Mean MEA: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
#results = search.fit(clean_df, labels)
results2 = lasso.fit(clean_df, labels)
print('Alpha: %.3f' % results2.alpha_)
#print('Config: %s' % results2.best_params_)