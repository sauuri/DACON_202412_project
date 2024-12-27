# %%
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
from sklearn.model_selection import GridSearchCV

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
            categorical_columns.append(col)
            print(categorical_columns)
    print('='*40)
    return categorical_columns

# FIXME !!!(2024-12-26-jhlee): Nan값도 encode 되어서 결측치를 따로 뽑을 수 없는 문제 발생
#                              check_unique_val_data 함수 object 관련 문제
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
    print(categorical_columns)
    data_copy = data.copy()
    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        data_copy[col] = encoder.fit_transform(data_copy[col])
        encoders[col] = encoder
        print(col)
    return data_copy, encoders
    
# %%
if __name__ == "__main__":
    # %%    
    train = pd.read_csv('./train.csv', index_col='ID')
    test = pd.read_csv('./test.csv', index_col='ID')

    # %%
    summarized_data(train, "Train Dataset")
    check_missing_data(train)
    categorical_columns = check_unique_val_data(train)
    train_data, encoders = label_encode_categorical_columns(train, categorical_columns)
    
    # TODO(2024-12-26-jhlee-1): 배터리 용량 예측 성능 키우기
    train_with_target = train_data[train_data['배터리용량'].notna()]
    
    # %%
    X_train = train_with_target.drop(columns=['배터리용량','가격(백만원)'])
    print(X_train)
    y_train = train_with_target['배터리용량']
    print(y_train)
    # %%
    test_without_target = train_data[train_data['배터리용량'].isna()]
    X_test = train_without_target.drop(columns=['배터리용량','가격(백만원)'])
    y_test = train_without_target['배터리용량']
    # %%
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 10],
        'n_estimators': [50, 100, 200, 300],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    battery_model = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=battery_model, param_grid=param_grid, cv=5, 
                                scoring='neg_root_mean_squared_error', verbose=2, n_jobs=20)
    grid_search.fit(X_train, y_train)
    
    # 최적의 파라미터 및 점수 확인
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Scoring: ", grid_search.best_score_)
    # %%
    best_params = grid_search.best_params_
    battery_final_model = xgb.XGBRegressor(**best_params, random_state=42)

    battery_final_model.fit(X_train, y_train)
    y_pred = battery_final_model.predict(X_train)

    rmse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"Validation RMSE: {rmse:.4f}")

    X_train_missing = train_without_target.drop(columns=['배터리용량', '가격(백만원)'])

    train_data.loc[train_data['배터리용량'].isna(), '배터리용량'] = battery_final_model.predict(X_train_missing)
    
    test_data, test_encoders = label_encode_categorical_columns(test, categorical_columns)

    X_test_missing = test_data[test_data['배터리용량'].isna()].drop(columns=['배터리용량'])
    test_data.loc[test['배터리용량'].isna(), '배터리용량'] = battery_final_model.predict(X_test_missing)
    
    X_train_final = train_data.drop(columns=['가격(백만원)'])
    y_train_final = train_data['가격(백만원)']
    
    X_test_final = test_data.copy()
    # TODO(2024-12-27, jhlee) -> 한번더 하이퍼 파라미터 튜닝 할지말지? 정하기
    final_model = xgb.XGBRegressor(**best_params, random_state=42)
    final_model.fit(X_train_final, y_train_final)
    test['예측_가격'] = final_model.predict(X_test_final)
    sample_file = pd.read_csv('./sample_submission.csv', index_col='ID')
    # %%
    sample_file['가격(백만원)'] = final_model.predict(X_test_final)
    # %%
    from datetime import datetime
    datetime.today() 
    # %%
    #sample_file.to_csv(f'sample_submission_{datetime.today().month}_{datetime.today().day}_{datetime.today().hour}_{datetime.today().minute}.csv')
# %%