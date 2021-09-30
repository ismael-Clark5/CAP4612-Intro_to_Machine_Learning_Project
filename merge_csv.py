#!/bin/python
import pandas as pd;
import os
import random as rd
import threading as th
import asyncio

training_df = pd.DataFrame();
testing_df = pd.DataFrame()
random = rd.Random()
threads = []

lock = asyncio.Lock()
async def loadToMem(df: pd.DataFrame, fileName: str):
    testing = pd.DataFrame()
    training =  pd.DataFrame()
    for i,row in df.iterrows():
        picked = random.randrange(0,100)
        if picked < 20:
            testing = testing.append(row)
        else:
            training = training.append(row)
    await lock.acquire()
    # Lock out other threads to prevent race conditions
    # Prvent data loss when overwritting global dataframe from multiple threads
    try: 
        global training_df
        global testing_df
        training_df = training_df.append(training)
        testing_df = testing_df.append(testing)
    finally:
        lock.release()
        print(f'File: {fileName} DONE!')

#Starts Here   
for file in os.listdir(os.getcwd()):
    if(file.endswith('.csv')):
        print(f'Working on File: {file}')
        file_df = pd.read_csv(file)
        print(f'Loaded Entries (Rows): {len(file_df.index)}')
        thread = th.Thread(target=lambda: asyncio.run(loadToMem(file_df, file)))
        threads.append(thread)
        thread.start()

#Join all threads
print('All Files Loaded, Waiting on threads!')
for thread in threads:
    thread.join()

#fill empty spaces              
print('Filling in empty spaces with "N/A"')        
training_df = training_df.fillna(value='na', axis=0,inplace=True)
testing_df = testing_df.fillna(value='na', axis=0,inplace=True)
#Write to CSV 
print(f'Training Data # Rows: {len(training_df.index)}')
training_df.to_csv('training_merged.csv')
print(f'Testing Data # Rows: {len(testing_df.index)}')
testing_df.to_csv('testing_merged.csv')
