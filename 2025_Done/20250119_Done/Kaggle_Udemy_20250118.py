import pandas as pd
import numpy as np

df = pd.read_csv('sample1_with_index.csv')
# print(df.head())
# print(df.info())
# print("*****")
# print(df.isnull().sum())
# print("*****")
print(df.dropna(subset=['Date', 'Color', 'Shape'])) # , inplace=True))