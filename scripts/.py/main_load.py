# --- START OF FILE main_load.py ---
from pathlib import Path
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

output_dir = Path(r'C:\\Users\\luke1\\Desktop\\projects\\T-brain-SinoPac-AIGO\\data\\public_dataset.csv')
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

def main():
    # asyncio.run(main_async())

    df = pd.read_csv(output_dir)

    df = df.rename(columns=data_columns)

    df_result = df['ID']
    df=df.drop(["ID", "Longitude","Latitude", "Street_Name", "Remarks"], axis=1)
    df['Address'] = df['City'] + df['District']
    df = df.drop(columns=['City', 'District'])

    # Label encoding
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df['Zoning'] = label_encoder.fit_transform(df['Zoning'])
    Zoning_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    df['Primary_Use'] = label_encoder.fit_transform(df['Primary_Use'])
    Primary_Use_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    df['Primary_Construction_Material'] = label_encoder.fit_transform(df['Primary_Construction_Material'])
    Primary_Construction_Material_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    df['Building_Type'] = label_encoder.fit_transform(df['Building_Type'])
    Building_Type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    df['Address'] = label_encoder.fit_transform(df['Address'])
    # Address_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


    model = Sequential()
    model = pickle.load(open(r'output\\output.pickle', "rb"))

    from sklearn.compose import ColumnTransformer
    numerical_cols = ['Land_Area', 'Age_of_Building', 'Building_Area', 'Parking_Area', 'Main_Building_Area', 'Balcony_Area', 'Auxiliary_Building_Area']
    categorical_cols = ['Zoning', 'Transfer_Level', 'Total_Floors', 'Primary_Use', 'Primary_Construction_Material', 'Building_Type', 'Number_of_Parking_Spaces', 'Address']

    # Apply label encoding to categorical columns
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    # Define column transformer to separately scale numerical and label-encoded categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', 'passthrough', categorical_cols)  # 'passthrough' indicates no transformation for categorical features
        ])

    # Fit and transform using the column transformer
    df = preprocessor.fit_transform(df)

    predictions = model.predict(df)
    predictions_column = pd.DataFrame(predictions, columns=['predicted_price'])
    df_result = pd.concat([df_result, predictions_column], axis=1)
    print(df_result)
    df_result.to_csv(r'output/result.csv', index=False)

if __name__ == "__main__":
    main()
# --- END OF FILE main_load.py ---