import json
import data as d

training_features, training_labels = d.get_training_split()
testing_features, testing_labels = d.get_test_split()

sfm_features = json.load(open('../Datasets/SFM.json', 'r'))
rfe_features = json.load(open('../Datasets/RFE.json', 'r'))
lasso_features  = json.load(open('../Datasets/LASSO.json', 'r')) 

sfm_dataframes = dict()
rfe_dataframes = dict()
lasso_dataframes = dict()

# Create a dictionary pointing from num of features to a dataframe containing only the columns with those feature names.
# Because rfe and sfm contain same number type of data, they can be done at the same time.
for num_features in sfm_features.keys():
    sfm_dataframes[num_features] = training_features.filter(items=sfm_features[num_features], axis=1)
    rfe_dataframes[num_features] = training_features.filter(items=rfe_features[num_features], axis=1)

# Do the same but with LASSO features.
for alpha in lasso_features.keys():
    lasso_dataframes[len(lasso_features[alpha])] = training_features.filter(items=lasso_features[alpha], axis=1)

print(lasso_dataframes)