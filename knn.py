import json
import ML_Models.data as d

training_festures, training_labels = d.get_train_split()
testing_features, testing_labels = d.get_test_split()

sfm_features = json.load(open('../Datasets/SFM.json', 'r'))
rfe_features = json.load(open('../Datasets/RFE.json', 'r'))
#lasso_features  = json.load(open('../Datasets/LASSO.json', 'r')) 

sfm_dataframes = dict()
rfe_dataframes = dict()
lasso_dataframes = dict()

# Create a dictionary pointing from num of features to a dataframe containing only the columns with those feature names.
# Because rfe and sfm contain same number type of data, they can be done at the same time.
for num_features in sfm_features.keys():
    sfm_dataframes[num_features] = training_festures.filter(items=sfm_features[num_features], axis=1)
    rfe_dataframes[num_features] = training_festures.filter(items=rfe_dataframes[num_features], axis=1)

# Do the same but with LASSO features.