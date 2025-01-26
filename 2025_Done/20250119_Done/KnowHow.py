import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# KnowHow.py

# 1. データ読み込み

def load_data(file_path):
    return pd.read_csv(file_path)

# 2. データの基本情報確認
def data_overview(df):
    print(df.info())
    print(df.describe())

# 3. 欠損値の確認と処理
def handle_missing_values(df):
    print(df.isnull().sum())
    df = df.dropna()  # 欠損値を含む行を削除
    return df

# 4. データの可視化
import matplotlib.pyplot as plt

def visualize_data(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.show()

# 5. データの前処理

def preprocess_data(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# 6. モデルの訓練と評価

def train_and_evaluate(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

# Example usage
if __name__ == "__main__":
    df = load_data('sample1_with_index.csv')
    data_overview(df)
    df = handle_missing_values(df)
    
    visualize_data(df, 'column_name')
    df = preprocess_data(df, ['column1', 'column2'])
    train_and_evaluate(df, 'target_column')
    # 7. モデルの保存と読み込み

    def save_model(model, file_path):
        joblib.dump(model, file_path)
        print(f'Model saved to {file_path}')

    def load_model(file_path):
        model = joblib.load(file_path)
        print(f'Model loaded from {file_path}')
        return model

    # Example usage for saving and loading the model
    save_model(model, 'linear_regression_model.pkl')
    loaded_model = load_model('linear_regression_model.pkl')