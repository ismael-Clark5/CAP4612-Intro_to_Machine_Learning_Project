import json

from seaborn.palettes import color_palette
import data as d
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


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


## Begin t-SENE
fig, plots = plt.subplots(2,3)
df, labels = d.get_training_split()
tsne = TSNE(n_components=3, verbose=0)
plotx = 0
ploty = 0
for numfeatures, df in sfm_dataframes.items():
    if ploty == 3:
        plotx += 1
        ploty = 0
    df_t = tsne.fit_transform(df)
    tsne_for_seaborn = pd.DataFrame(data=df_t, columns=['tSNE1','tSNE2', 'tSNE3'])
    plots[plotx, ploty].set_title(f'SFM t-SNE with {numfeatures}')
    sns.scatterplot(
        x="tSNE1",y="tSNE2",
        data=tsne_for_seaborn,
        hue=labels,
        legend="full",
        palette=sns.color_palette('hls', 12),
        ax=plots[plotx, ploty]
    )
    ploty += 1

plt.show()

fig, plots = plt.subplots(2,3)
plotx = 0
ploty = 0
for numfeatures, df in rfe_dataframes.items():
    if ploty == 3:
        plotx += 1
        ploty = 0
    df_t = tsne.fit_transform(df)
    tsne_for_seaborn = pd.DataFrame(data=df_t, columns=['tSNE1','tSNE2'])
    plots[plotx, ploty].set_title(f'RFE t-SNE with {numfeatures}')
    sns.scatterplot(
        x="tSNE1",y="tSNE2",
        data=tsne_for_seaborn,
        hue=labels,
        legend="full",
        palette=sns.color_palette('hls', 12),
        ax=plots[plotx, ploty]
    )
    ploty += 1
plt.show()

fig, plots = plt.subplots(2, 2)
plotx = 0
ploty = 0
counter = 0
for numfeatures, df in lasso_dataframes.items():
    if plotx == 2:
        ploty += 1
        plotx = 0
    if counter == 4:
        plt.show()
        fig, plots = plt.subplots(2, 2)
        plotx = 0
        ploty = 0
        counter = 0
    df_t = tsne.fit_transform(df)
    tsne_for_seaborn = pd.DataFrame(data=df_t, columns=['tSNE1','tSNE2'])
    plots[plotx, ploty].set_title(f'LASSO t-SNE with {numfeatures} features')
    sns.scatterplot(
        x="tSNE1",y="tSNE2",
        data=tsne_for_seaborn,
        hue=labels,
        legend="full",
        palette=sns.color_palette('hls', 12),
        ax=plots[plotx, ploty]
    )
    plotx += 1
    counter += 1
plt.show()
