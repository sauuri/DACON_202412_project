import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 한글 폰트 사용 위함
rc('font', family='Malgun Gothic')


def summarized_data(data, data_name="Dataset"):
    """
    데이터 구조를 요약 출력하는 함수.

    Args:
        data (pd.DataFrame): 요약할 데이터프레임
        name (str): 데이터 이름 (출력 시 사용)
    """
    # 데이터 구조 확인
    print("=" * 40, f'\n{data_name} information (.info)\n{"=" * 40}')
    data.info() # .info는 자체 출력

    print("=" * 40, f'\n{data_name} Summary Statistics (.describe)\n{"=" * 40}')
    print(data.describe(),'\n')

    print("=" * 40, f'\n{data_name} (First 5 Rows - .head())\n{"=" * 40}') 
    print(data.head(),'\n')

    print("=" * 40, f'\n{data_name} Shape (.shape)\n{"=" * 40}')
    print(f'Rows: {data.shape[0]}, Columns: {data.shape[1]}\n{"=" * 40}')

def check_missing_data(data):
    """
    데이터의 결측치를 확인하는 함수.

    Args:
        data(pd.DataFrame): 결측치를 확인할 데이터프레임
    """
    msno.matrix(data)
    #plt.show()

def check_unique_val_data(data):
    """
    데이터프레임에서 범주형 열의 고유값을 출력하는 함수.

    Args:
        data (pd.DataFrame): 데이터프레임
    """

    print('='*40)
    
    categorical_columns = []
    for col in data.columns:
        if data[col].dtype == object:
            print(f"Column: {col}")
            print(f"Unique values: {data[col].unique()}\n")
        categorical_columns.append(data[col])
    
    print('='*40)
    return categorical_columns

def label_encode_categorical_columns(data, categorical_columns):
    """
    범주형 열을 정수형으로 변환하고, 변환된 데이터프레임과 LabelEncoder 객체를 반환.

    Args:
        data (pd.DataFrame): 변환할 데이터프레임
        categorical_columns: 변환할 범주형 열의 리스트

    Returns:
        pd.DataFrame: 변환된 데이터프레임
        dict: 각 열에 대한 LabelEncoder 객체의 딕셔너리
    """
    data_copy = data.copy()
    encoders = {}
    for col in data.columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder
    return data, encoders

#[DEBUG] 
# TODO

def except_battery():
    pass
    return 

if __name__ == "__main__":
    
    train = pd.read_csv('./train.csv', index_col='ID')
    test = pd.read_csv('./test.csv', index_col='ID')

    summarized_data(train, "Train Dataset")
    check_missing_data(train)
    categorical_columns = check_unique_val_data(train)    
    train, encoders = label_encode_categorical_columns(train, categorical_columns)

    train_with_target = train[train['배터리용량'].notna()]
    X_train = train_with_target.drop(columns=['배터리용량','가격(백만원)'])
    y_train = train_with_target['배터리용량']

    train_without_target = train[train['배터리용량'].isna()]

    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    battery_model = xgb.XGBRegressor(random_state=42)
    battery_model.fit(X_train_split, y_train_split)

    y_valid_pred = battery_model.predict(X_valid_split)

    rmse = mean_squared_error(y_valid_split, y_valid_pred, squared=False)
    print(f"Validation RMSE: {rmse:.4f}")

    X_train_missing = train_without_target.drop(columns=['배터리용량'])
    train.loc[train['배터리용량'].isna(), '배터리용량'] = battery_model.predict(X_train_missing)
    
    X_test_missing = test[test['배터리용량'].isna()].drop(columns=['배터리용량'])
    test.loc[test['배터리용량'].isna(), '배터리용량'] = battery_model.predict(X_test_missing)

    X_train_final = train.drop(columns=['가격(백만원)'])
    y_train_final = train['가격(백만원)']

    final_model = xgb.XGBRegressor(random_state=42)
    final_model.fit(X_train_final, y_train_final)

    test['예측_가격'] = final_model.predict(test)
