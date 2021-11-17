import json
from types import ModuleType

from seaborn.palettes import color_palette
import data as d
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm  import SVC as svm 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier as ran_forest
import scikitplot as skplt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
def tsne():
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

def classification(classifier, data_frames, df_name) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    scores = dict()
    print(f"Begin Classification for {classifier.__class__.__name__} with Dataframes from {df_name}")
    for num_features, df in data_frames.items(): 
        scores[num_features] = []
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        predicted_values = []
        ground_values = []
        probability_values = []
        y_tests = np.empty(shape=(0))
        y_probabilities = np.empty(shape=(0,12))
        for train_index, test_index in skf.split(df, training_labels):
            X_train, X_test = df.iloc[train_index], df.iloc[test_index]
            y_train, y_test = training_labels.iloc[train_index], training_labels.iloc[test_index]
            classifier.fit(X_train, y_train)
            prediction=classifier.predict(X_test)
            predicted_values = np.append(predicted_values, prediction)
            ground_values = np.append(ground_values, y_test)
            y_tests = np.append(y_tests, y_test, axis=0)
            probability = classifier.predict_proba(X_test)
            y_probabilities = np.append(y_probabilities, probability, axis=0)
            probability_values = np.append(probability_values, probability)
            accuracies.append(accuracy_score(y_test, prediction, sample_weight=None, normalize=True))
            precisions.append(precision_score(y_test, prediction, sample_weight=None, average='micro'))
            recalls.append(recall_score(y_test, prediction, sample_weight=None, average='micro'))
            f1s.append(f1_score(y_test, prediction, sample_weight=None, average='micro'))  
        cm = confusion_matrix(ground_values, predicted_values)
        skplt.metrics.plot_roc(y_tests, y_probabilities, title=f'{classifier.__class__.__name__} with {num_features} features (from {df_name})')
        plt.show()
        mean_accuracy = np.mean(accuracies)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1 = np.mean(f1s)
        print(f'Num of Features: {num_features},\nMean Accuracy: {mean_accuracy},\nMean Precision: {mean_precision},\nMean Recall: {mean_recall},\nMean F1: {mean_f1}')
    return scores


classifiers = [KNeighborsClassifier(n_neighbors=5), svm(kernel='linear', probability=True), ran_forest(n_estimators=100)]
data_frames = [sfm_dataframes, lasso_dataframes, rfe_dataframes]

if __name__ == '__main__':
    scores = dict()
    for classifier in classifiers:
        scores[classifier.__class__.__name__] = [("SFM", classification(classifier, sfm_dataframes, "SFM"))]
        scores[classifier.__class__.__name__].append(["LASSO",classification(classifier, lasso_dataframes, "LASSO")])
        scores[classifier.__class__.__name__].append(["RFE", classification(classifier, rfe_dataframes, "RFE")])
    #print(scores)

    """
    confusion matrix, mean accuracy, mean precision, mean recall, mean f1 
    score, ROC curve, AUC score.
    """