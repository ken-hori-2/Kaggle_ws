# import pandas as pd
# import numpy as np
# import seaborn as sns
# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.metrics import mean_squared_error
# import lightgbm as lgb
# import xgboost as xgb
# from sklearn.ensemble import RandomForestRegressor
# import warnings

# # knowhow_main.py

# import matplotlib.pyplot as plt
# warnings.filterwarnings('ignore')

# # 1. データ読み込み
# def load_data(file_path):
#     return pd.read_csv(file_path)

# # 2. データの確認
# def check_data(df):
#     print(df.head())
#     print(df.info())
#     print(df.describe())
#     print(df.isnull().sum())

# # 3. 探索的データ分析(可視化や統計分析)
# def exploratory_data_analysis(df):
#     sns.pairplot(df)
#     plt.show()
#     sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
#     plt.show()

# # 4. データの前処理(欠損値処理、特徴量作成)
# def preprocess_data(df):
#     df.fillna(df.mean(), inplace=True)
#     # 特徴量作成の例
#     df['new_feature'] = df['feature1'] * df['feature2']
#     return df

# # 5. 学習(LightGBM, XGBoost, NeuralNetworkおよび、アンサンブル学習)
# def train_model(df, target_column):
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     models = {
#         'LightGBM': lgb.LGBMRegressor(),
#         'XGBoost': xgb.XGBRegressor(),
#         'RandomForest': RandomForestRegressor()
#     }
    
#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         print(f'{name} RMSE: {rmse}')

# # 6. 評価(kFold)
# def evaluate_model(df, target_column):
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
    
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
#     model = lgb.LGBMRegressor()
#     scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
#     rmse_scores = np.sqrt(-scores)
#     print(f'KFold RMSE: {rmse_scores.mean()}')

# # 7. csv形式でkaggleの提出形式で出力
# def output_submission(df, model, file_path):
#     predictions = model.predict(df)
#     submission = pd.DataFrame({'Id': df.index, 'Prediction': predictions})
#     submission.to_csv(file_path, index=False)

# # 使用例
# if __name__ == "__main__":
#     file_path = 'sample1_with_index.csv'
#     df = load_data(file_path)
#     check_data(df)
#     exploratory_data_analysis(df)
#     df = preprocess_data(df)
#     train_model(df, 'target_column')
#     evaluate_model(df, 'target_column')
#     # モデルを再学習して提出用ファイルを出力
#     model = lgb.LGBMRegressor()
#     model.fit(df.drop(columns=['target_column']), df['target_column'])
#     output_submission(df, model, 'submission.csv')


# # あなたは優秀なデータ分析エンジニアです。これさえ見ればあら有るテーブルデータ(kaggleなどのコンペ)に対応できるというチートシート(処理の手順)をまとめて教えてください。
# # 例：
# # 1. データ読み込み
# # 2. データの確認
# # 3. 探索的データ分析(可視化や統計分析)
# # 4.データの前処理(欠損値処理、特徴量作成)
# # 5. 学習(LightGBM, XGBoost, NeuralNetworkおよび、アンサンブル学習)
# # 6. 評価(kFold)
# # 7. csv形式でkaggleの提出形式で出力


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 1. データ読み込み
def load_data(file_path):
    return pd.read_csv(file_path)

# 2. データの確認
def check_data(df):
    print(df.head())
    # print(df.info())
    # print(df.describe())
    # print(df.isnull().sum())
    
    # データの概要を確認
    print(df.info())
    # 統計情報の確認
    print(df.describe())
    # カラムごとの欠損値の確認
    print(df.isnull().sum())
    # データ型の確認
    print(df.dtypes)


# 3. 探索的データ分析(可視化や統計分析)
def exploratory_data_analysis(df):
    # sns.pairplot(df)
    # plt.show()
    # sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    # plt.show()
    
    # 目的変数の分布
    sns.countplot(x='target', data=df)
    plt.show()
    # 数値データの相関行列を可視化
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.show()
    # カテゴリデータの分布
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        sns.countplot(x=col, data=df)
        plt.show()
    # 異常値の検出（箱ひげ図）
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

# 4. データの前処理(欠損値処理、特徴量作成)
def preprocess_data(df):
    # df.fillna(df.mean(), inplace=True)
    # 特徴量作成の例
    # df['new_feature'] = df['feature1'] * df['feature2']
    
    # 4.1 欠損値処理
    # 平均値で補完
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    # 最頻値で補完
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    # 欠損値の削除
    df.dropna(inplace=True)
    
    # 4.2 特徴量エンジニアリング
    # 新しい特徴量の作成
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # カテゴリデータのエンコーディング
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    # 不要なカラムを削除
    df.drop(['Name', 'Ticket'], axis=1, inplace=True)
    
    # 4.3 データスケーリング
    scaler = StandardScaler()
    numerical_cols = ['Age', 'Fare']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df

# 5. 学習(LightGBM, XGBoost, NeuralNetworkおよび、アンサンブル学習)
def train_model(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'LightGBM': lgb.LGBMRegressor(),
        'XGBoost': xgb.XGBRegressor(),
        'RandomForest': RandomForestRegressor()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f'{name} RMSE: {rmse}')

# 6. 評価(kFold)
def evaluate_model(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    model = lgb.LGBMRegressor()
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
    rmse_scores = np.sqrt(-scores)
    print(f'KFold RMSE: {rmse_scores.mean()}')

# 7. csv形式でkaggleの提出形式で出力
def output_submission(df, model, file_path):
    predictions = model.predict(df)
    submission = pd.DataFrame({'Id': df.index, 'Prediction': predictions})
    submission.to_csv(file_path, index=False)

# 使用例
if __name__ == "__main__":
    file_path = 'sample1_with_index.csv'
    df = load_data(file_path)
    check_data(df)
    exploratory_data_analysis(df)
    df = preprocess_data(df)
    train_model(df, 'target_column')
    evaluate_model(df, 'target_column')
    # モデルを再学習して提出用ファイルを出力
    model = lgb.LGBMRegressor()
    model.fit(df.drop(columns=['target_column']), df['target_column'])
    output_submission(df, model, 'submission.csv')