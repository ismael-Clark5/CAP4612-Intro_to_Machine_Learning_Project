import pandas as pd
from sklearn.model_selection import train_test_split

training_df = pd.read_csv("../Datasets/training_merged.csv")
testing_df = pd.read_csv("../Datasets/testing_merged.csv")

def getSplits():
    unified_df = training_df.append(testing_df)
    X_train, X_test, Y_train, Y_test = train_test_split(unified_df, test_size=0.2)
    return (X_train, X_test, Y_train, Y_test)