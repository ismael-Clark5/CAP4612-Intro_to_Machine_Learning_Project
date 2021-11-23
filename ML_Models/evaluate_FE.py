import json

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
    """
        Function handles the plotting of the t-SNE for the diffrent algorithms and the number of features selected in those algorithms
    """
    fig, plots = plt.subplots(2,3)
    df, labels = d.get_training_split()
    tsne = TSNE(n_components=2, verbose=0)
    plotx = 0
    ploty = 0
    for numfeatures, df in sfm_dataframes.items():
        if ploty == 3:
            plotx += 1
            ploty = 0
        df_t = tsne.fit_transform(df)
        tsne_for_seaborn = pd.DataFrame(data=df_t, columns=['tSNE1','tSNE2'])
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


def classification(classifier, data_frames) -> tuple[dict, dict, dict, dict, dict, dict]:
    """
    Function handles the classification proccess and extraction of the information from the classification process
    The classification proccess if verified using Stratified K Fold algorithm with 5 splits per run. 
    This process is run per number of features selected on the previous steps. 
    The values for accuracy, percision, recall, f1, and the values for roc curve are then stored in a dictionary, with the mapping of, 
    number_features -> data structure containing values for the section, (typically list, or numpy array)

    @param: classifier: Classifier to be used in the classification process.
    @param: data_frames: dictionary containing mapping num_features -> dataframe containing only those features. 
    @return: Tuple[accuracy, precision, recall, f1, cm, roc_values]
    """

    #Create stratified kfold object 
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    #Define dictinaries to collect results. 
    accuracy_dict = dict()
    precision_dict = dict()
    recall_dict = dict()
    f1_dict = dict()
    cm_dict = dict()
    roc_values = dict()

    #Iterate over the number of features and dataframes in the provided data_frames dictionary
    for num_features, df in data_frames.items(): 
        #Declare variables to store values for this number of features. 
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        predicted_values = []
        ground_values = []
        probability_values = []
        y_tests = np.empty(shape=(0))
        y_probabilities = np.empty(shape=(0,12))

        #Start iterating over the stratified Kfold splits. 
        for train_index, test_index in skf.split(df, training_labels):
            #Extract selected samples for training and testing. 
            X_train, X_test = df.iloc[train_index], df.iloc[test_index]
            y_train, y_test = training_labels.iloc[train_index], training_labels.iloc[test_index]
            #Train the model
            classifier.fit(X_train, y_train)
            #Predict the values on test 
            prediction=classifier.predict(X_test)
            #Start Extracting the values form the classifier. 
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

        # Store the values from the stratified k fold runs into the dictionary for the number of features. 
        cm_dict[num_features] = confusion_matrix(ground_values, predicted_values)
        accuracy_dict[num_features] = np.mean(accuracies)
        precision_dict[num_features] =  np.mean(precisions)
        recall_dict[num_features] = np.mean(recalls)
        f1_dict[num_features] = np.mean(f1s)
        roc_values[num_features] = (y_tests, y_probabilities)

    #Return the values in tuple format 
    return (accuracy_dict, precision_dict, recall_dict, f1_dict, cm_dict, roc_values)

def plot_section(classification_data: tuple[dict, dict, dict, dict, dict, dict], model_name: str, classifier_name: str ):
    """
        This function takes care of creating the plots for the data geathered in the classification step. 
        The data in the passed tuple is in the mapping of num_features -> data structure containing values (typically list or numpy arrays)
        @param: classification_data: tuple[dict, dict, dict, dict, dict, dict], contains dictionaries for accuracy, precision, recall, f1, confusion matrix, and roc
        @param: model_name: string,  Model which was used to extract the features from the full dataset. 
        @classifier_name: string, Classifier model used to classify the features. 
    """

    # Extract values from the classification data tuple object. 
    accuracy_dict, precision_dict, recall_dict, f1_dict, cm_dict, roc_values_dict = classification_data

    #Deine the sections which can be represented together 
    sections = {
        "Accuracy" : accuracy_dict,
        "Precision" : precision_dict,
        "Recall" : recall_dict,
        "F1": f1_dict
    }
    #Create the plots and start plotting the sections. 
    fig, plots = plt.subplots(2,2) 
    plotx = 0
    ploty = 0
    for section, dictionary in sections.items():
        if plotx == 2:
            plotx = 0
            ploty += 1
        x, y = zip(*(dictionary.items()))
        plots[plotx, ploty].plot(x, y)
        plots[plotx, ploty].set_title(f"Mean {section} ({model_name}) using {classifier_name} vs Number of Features")
        plotx += 1 
    plt.show()

    # Confision Matrix needs to have its own plot, as we have a confusion matrix per each number of features. 
    # This section takes care of plotting each confusion matrix for the features. 
    fig, plots = plt.subplots(2, (len(cm_dict)//2) + 1)
    plotx = 0
    ploty = 0
    if( model_name == "LASSO"):
        plot_lasso(classification_data, classifier_name)
        return
    else:
        for num_features, confusion_matrix in cm_dict.items():
            if plotx == 2:
                plotx = 0
                ploty += 1
            #sns.heatmap(confusion_matrix, annot=True, ax=plots[plotx, ploty], robust=True)
            plots[plotx, ploty].matshow(confusion_matrix, cmap='binary')
            plots[plotx, ploty].set_title(f"CM ({model_name}) for {classifier_name} with {num_features} Features")
            plotx +=1 
        plt.show()

    # Like confusion matrix, ROC curves are drawn per number of features and thus must be plotted seperatly. 
    fig, plots = plt.subplots(2, (len(roc_values_dict)//2) + 1)
    plotx = 0
    ploty = 0
    for num_features, (y_test, y_proba) in roc_values_dict.items():
        if plotx == 2:
            plotx = 0
            ploty += 1
        skplt.metrics.plot_roc(y_test, y_proba, title=f'ROC ({model_name}) {classifier_name} with {num_features} Features', ax=plots[plotx, ploty] )
        plotx +=1 
    plt.show()


def plot_lasso(classification_data: tuple[dict, dict, dict, dict, dict, dict] , classifier_name: str):
    """
    Due to how lasso has many, many features to plot, it is easier to have them be plotted seperetly
    where the number of plots is designs specifically for LASSO.
    @param: cm_dict: Dictionary containing a mapping of int-> dict, where int is the number of features and dict is the values used to create the confusion matrix for tha many features
    @param: roc_dict: Dictionary containing mappig of int-> tuple[y_test, y_proba], where in the the number of features and tuple containts the set of values required to create an ROC curve for that number fo features
    @param: classifier_name: String containg the classifier class used to test the feautures 
    """
    _, __, ___, ___, cm_dict, roc_values_dict = classification_data

    fig, plot = plt.subplots(2,2)
    plotx = 0
    ploty = 0
    for num_features, confusion_matrix in cm_dict.items():
        if plotx == 2 and ploty == 1:
            fig, plot = plt.subplots(2,2)
            plotx = 0
            ploty = 0
        if plotx == 2:
            ploty += 1
            plotx = 0
        plot[plotx, ploty].matshow(confusion_matrix, cmap="binary")
        plot[plotx, ploty].set_title(f"CM LASSO for {classifier_name} with {num_features} Features")
        plotx+=1
    plt.show()

    fig, plot = plt.subplots(2,2)
    plotx = 0
    ploty = 0
    for num_features in roc_values_dict.keys():
        y_test, y_proba = roc_values_dict[num_features]
        if plotx == 2 and ploty == 1:
            fig, plot = plt.subplots(2,2)
            plotx = 0
            ploty = 0
        if plotx == 2:
            ploty += 1
            plotx = 0
        skplt.metrics.plot_roc(y_test, y_proba, title=f'ROC (LASSO) {classifier_name} with {num_features} Features', ax=plot[plotx, ploty] )
        plotx+=1
    plt.show()
        

    

classifiers = [KNeighborsClassifier(n_neighbors=5), svm(kernel='linear', probability=True), ran_forest(n_estimators=100)]
data_frames = [sfm_dataframes, lasso_dataframes, rfe_dataframes]

if __name__ == '__main__':
    tsne()
    for classifier in classifiers:
        classification_output_sfm = classification(classifier, sfm_dataframes)
        plot_section(classification_output_sfm, "SFM", classifier.__class__.__name__)
        classification_output_lasso = classification(classifier, lasso_dataframes)
        plot_section(classification_output_lasso, "LASSO", classifier.__class__.__name__)
        classification_output_rfe = classification(classifier, rfe_dataframes)
        plot_section(classification_output_rfe, "RFE", classifier.__class__.__name__)
    #print(scores)

    """
    confusion matrix, mean accuracy, mean precision, mean recall, mean f1 
    score, ROC curve, AUC score.
    """