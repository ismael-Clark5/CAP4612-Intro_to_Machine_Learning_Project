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
def getSplits():
    unified_df = training_df.append(testing_df)
    X_train, X_test, Y_train, Y_test = train_test_split(unified_df, test_size=0.2)
    return (X_train, X_test, Y_train, Y_test)