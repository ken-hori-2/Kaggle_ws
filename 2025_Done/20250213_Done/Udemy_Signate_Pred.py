import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_submit = pd.read_csv('sample_submission.csv', header=None, names=['Date', 'Up'])

# print(df_train)
# print("="*20)
# print(df_test)
# print("="*20)
# print(df_submit) # このままだとおかしいのでheaderを指定

print(df_train.head())
print("="*20)

# pd.concat([df_train['Close'], df_test['Close']], axis=0).plot() # このままだとIndexがおかしいので、以下のように処理する
concated_data = pd.concat([df_train['Close'], df_test['Close']], axis=0).reset_index(drop=True).plot()
print(concated_data)
plt.show()


"""
このままモデル構築すると、株価が低いところで学習し（trainデータ）、株価が高いところで予測する（testデータ）ので、モデルがうまく動かないと考えられる。
なので、直接値を扱うのではなく、差分を扱う

時系列データでは、差分を取ることが重要
"""
print(df_train['Close'].diff(1)) # Closeの1日の差分（1日前との差分）

concated_data = pd.concat([df_train['Close'].diff(1), df_test['Close'].diff(1)], axis=0).reset_index(drop=True).plot()
print(concated_data)
plt.show()

# 次回
# 30. 特徴量エンジニアリング：ラグ特徴量を作ってみよう！