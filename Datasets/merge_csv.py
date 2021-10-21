#!/bin/python
from typing import List
import pandas as pd;
import os
import random as rd
from sklearn.model_selection import train_test_split

training_df = pd.DataFrame();
testing_df = pd.DataFrame()
random = rd.Random()

def loadToMem(df: pd.DataFrame, cancer: str):
    df.insert(1, 'cancer', cancer)
    training, testing = train_test_split(df, train_size=0.8, test_size=0.2)

    global training_df
    global testing_df
    training_df = training_df.append(training)
    testing_df = testing_df.append(testing)
    print(f'Cancer: {cancer} DONE!')

count = 0
#Starts Here
for file in os.listdir(os.getcwd()):
    if(file.endswith('.csv')):
        cancer = file.split('_')[1]
        file_df = pd.read_csv(file)
        print(f'Cancer: {cancer}, # Rows {len(file_df.index)}')
        count += len(file_df.index)
        loadToMem(file_df, cancer)

print(f'Total Rows Accross all Files: {count}')
print(f'Training Data # Rows: {len(training_df.index)}')
print(f'Testing Data # Rows: {len(testing_df.index)}')

#Write to CSV
training_df.to_csv('training_merged.csv')
testing_df.to_csv('testing_merged.csv')
print('All Files Written')