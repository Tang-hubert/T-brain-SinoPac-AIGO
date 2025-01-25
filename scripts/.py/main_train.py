# --- START OF FILE main_train.py ---
import asyncio
import datetime
import shutil
import tempfile
import warnings
from pathlib import Path

from absl import app, flags, logging

import pandas as pd
import seaborn as sns
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, LeakyReLU, Dropout

# Main
data_dir = Path(r'C:\\Users\\luke1\\Desktop\\projects\\T-brain-SinoPac-AIGO\\data\\training_data.csv')
data_columns = {
    'ID': 'ID',
    '縣市': 'City',
    '鄉鎮市區': 'District',
    '路名': 'Street_Name',
    '土地面積': 'Land_Area',
    '使用分區': 'Zoning',
    '移轉層次': 'Transfer_Level',
    '總樓層數': 'Total_Floors',
    '主要用途': 'Primary_Use',
    '主要建材': 'Primary_Construction_Material',
    '建物型態': 'Building_Type',
    '屋齡': 'Age_of_Building',
    '建物面積': 'Building_Area',
    '車位面積': 'Parking_Area',
    '車位個數': 'Number_of_Parking_Spaces',
    '橫坐標': 'Longitude',
    '縱坐標': 'Latitude',
    '備註': 'Remarks',
    '主建物面積': 'Main_Building_Area',
    '陽台面積': 'Balcony_Area',
    '附屬建物面積': 'Auxiliary_Building_Area',
    '單價': 'Unit_Price'
    }

# async def main_async():
#     print('hi')

def main():
    df = pd.read_csv(data_dir)

    df = df.rename(columns=data_columns)

    df=df.drop(["ID", "Longitude","Latitude", "Street_Name", "Remarks"], axis=1)
    df['Address'] = df['City'] + df['District']
    df = df.drop(columns=['City', 'District'])


    # to get each pairplot
    # pairplot(df)

    # Label encoding
    # from sklearn.preprocessing import LabelEncoder

    # label_encoder = LabelEncoder()
    # df['Zoning'] = label_encoder.fit_transform(df['Zoning'])
    # Zoning_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    # df['Primary_Use'] = label_encoder.fit_transform(df['Primary_Use'])
    # Primary_Use_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    # df['Primary_Construction_Material'] = label_encoder.fit_transform(df['Primary_Construction_Material'])
    # Primary_Construction_Material_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    # df['Building_Type'] = label_encoder.fit_transform(df['Building_Type'])
    # Building_Type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    # df['Address'] = label_encoder.fit_transform(df['Address'])
    # Address_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # print(Zoning_mapping)



    # seperate dataset
    X = df.drop(columns=['Unit_Price'])
    y = df['Unit_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # method
    # train(X_train, X_test, y_train, y_test)
    # call(X_test, y_test)

    # Assuming X_train, Y_train, X_test, Y_test are your training and testing data
    model = train_and_evaluate(X_train, y_train, X_test, y_test)

    pickle.dump(model, open(r'output\\output.pickle', "wb"))

    # print(Y_pred)
if __name__ == "__main__":
    main()

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def build_model():
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                         max_iter=1000, random_state=42, alpha=0.001)
    return model

def train_and_evaluate(X_train, Y_train, X_test, Y_test):
    from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # Assuming X_train and X_test contain both numerical and categorical features

    # Separate numerical and categorical columns
    numerical_cols = ['Land_Area', 'Age_of_Building', 'Building_Area', 'Parking_Area', 'Main_Building_Area', 'Balcony_Area', 'Auxiliary_Building_Area']
    categorical_cols = ['Zoning', 'Transfer_Level', 'Total_Floors', 'Primary_Use', 'Primary_Construction_Material', 'Building_Type', 'Number_of_Parking_Spaces', 'Address']

    # Apply label encoding to categorical columns
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        X_train[col] = label_encoders[col].fit_transform(X_train[col])
        X_test[col] = label_encoders[col].transform(X_test[col])

    # unknown_label = -1
    # for feature in categorical_cols:
    #     # 检查测试集中的标签是否在训练集的标签中
    #     unknown_labels = set(X_test[feature]) - set(label_encoders[feature].classes_)

    #     # 将未知标签映射为特定的未知标签值（例如，-1）
    #     for label in unknown_labels:
    #         X_test[feature][X_test[feature] == label] = unknown_label

    #     # 使用LabelEncoder进行转换
    #     X_test[feature] = label_encoders[feature].transform(X_test[feature])

    # Create a pipeline with scaling, polynomial feature generation, and label encoding
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('poly', PolynomialFeatures(degree=2, include_bias=False), numerical_cols),
                ('cat', 'passthrough', categorical_cols)  # 'passthrough' indicates no transformation for categorical features
            ])
        )
    ])

    # Fit and transform X_train using the pipeline
    X_train_processed = pipeline.fit_transform(X_train)

    # Transform X_test using the fitted pipeline
    X_test_processed = pipeline.transform(X_test)


    print(X_train_processed)


    # build
    model = build_model()
    model.fit(X_train_processed, Y_train)  # Fit the model with scaled training data

    Y_pred_train = model.predict(X_train_processed)
    mse_train = mean_squared_error(Y_train, Y_pred_train)
    mae_train = mean_absolute_error(Y_train, Y_pred_train)

    print("MSE_train: ", mse_train)
    print("MAE_train: ", mae_train)

    Y_pred_test = model.predict(X_test_processed)
    mse_test = mean_squared_error(Y_test, Y_pred_test)
    mae_test = mean_absolute_error(Y_test, Y_pred_test)

    print("MSE_test: ", mse_test)
    print("MAE_test: ", mae_test)

    # Cross-Validation Score
    cv_scores = cross_val_score(model, X_test_processed, Y_train, cv=5, scoring='neg_mean_squared_error')
    print("Cross-Validation MSE: ", -np.mean(cv_scores))

    return model

def paitplot(df2):
    res = df2[~df2.duplicated('City')]['City']
    res = res.reset_index(drop=True)
    print(res)

    for i in range(len(res)):
        print(df2[df2['City'] == res[i]].shape)
        plot = sns.pairplot(df2[df2['City'] == res[i]])
        plot.savefig(f"output/{res[i]}.png")
# --- END OF FILE main_train.py ---