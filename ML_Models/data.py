import pandas as pd
from sklearn.model_selection import train_test_split

training_df = pd.read_csv("../Datasets/training_merged.csv")
testing_df = pd.read_csv("../Datasets/testing_merged.csv")

cancer_to_int = {
    'BRCA': 1,
    'CHOL' : 2,
    'COAD' : 3,
    'KICH' : 4,
    'KIRC' : 5,
    'KIRP' : 6,
    'LIHC' : 7,
    'LUAD' : 8,
    'LUSC' : 9,
    'PRAD' : 10,
    'READ' : 11,
    'THCA' : 12
}
int_to_cancer = {v: k for k, v in cancer_to_int.items()}


def get_training_split():
    y = training_df['cancer']
    x = training_df.drop(labels=['Ensembl_ID', 'cancer', 'Unnamed: 0'], axis=1)

    return x,y


def get_test_split():
    y = testing_df['cancer']
    x = testing_df.drop(labels=['Ensembl_ID', 'cancer', 'Unnamed: 0'], axis=1)

    return x, y
