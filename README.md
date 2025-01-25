# T-brain-SinoPac-AIGO
## Done
- Have to fix the `y contains previously unseen labels` error.

## TO-DO
- Remember if you separate the predict function, when facing different unknown labels the LabelEncoding will change. -> Have to code it in the same file.(main)
- Lots of external_data try to use `Longitude` and `Latitude` and calculate each different installations and put (0,1) to recognized is it nearby or not.(Some are -1, have to ignore.)
- Execute some analysis from `pairplot_output.png`
    - Building_Area: against different Cities.
    - Age_of_Building: have to times -1 (Note: This likely means to multiply by -1 to correct the age, assuming age is stored as negative or needs to be reversed)
    - Parking_Area: Most are 0, and if it has won't effect the result (0,1) (Note: "won't effect" likely means "won't affect")
    - Auxiliary_Building_Area: Most are 0, and if it has won't effect the result (0,1) (Note: "won't effect" likely means "won't affect")
- Knowing how to evaluate the ML result.


# Good performance example: Yuanta Turing team approach sharing
> From [Sheng](https://hackmd.io/@shengchichi23/AI-GO-Turing-team)
## Mainly divided into Approach A and Approach B, and finally, the predictions of the two approaches are weighted sum.
The differences in feature engineering, model settings, and training methods between the two approaches are introduced separately below.

## Feature engineering
In this competition, we mainly focused on collecting external data and using real estate transaction price data.
The common and different parts of Approach A and B are introduced below.

### Common parts of Approach A and B
- Convert the latitude and longitude (WGS84) of each data to TWD97 coordinates.
- Divide "金融機構基本資料.csv" (Basic Data of Financial Institutions.csv) into "銀行.csv" (Bank.csv), "農會.csv" (Farmers' Association.csv), and "合作社.csv" (Cooperative.csv) through the "金融機構名稱" (Financial Institution Name) field.
- Add multiple additionally downloaded external data, including:
  - Substations
  - Night Markets
  - Parks
  - Power Plants
  - Recycling Companies
  - Gas Stations
  - Taipei City Public Cemetery
  - Funeral Facilities
  - Libraries
  - Landfills
  - Temples
  - Sewage Treatment Plants
  - Train Station Points
  - Incineration Plants
  - Police Stations
  - Post Offices
- Calculate using Manhattan distance:
  - The shortest distance between each house and each external data point.
  - Within 1000 meters, whether there is an MRT station providing bicycle rental service on holidays.
  - Within 1000 meters, the best train station level.
- Calculate using Euclidean distance:
  - The number of each external data point within X meters; X is searched by hyperopt, and the metric is to maximize the absolute value of the correlation between the calculated number and the label.
- Use real estate transaction price data from 2021 to 2022 to predict the real estate transaction price of the competition data as an additional feature column.
  - Use fields that overlap with the competition data during training.
  - Map categorical fields to the same categories as the competition data as much as possible.
  - Standardize numerical fields so that the mean and standard deviation are consistent with the competition data.
  - Train using autogluon, and the parameters used are as follows:
    - `time_limit=86400`
    - `presets="best_quality"`
  - The correlation between the predicted real estate transaction price and the label on the training data is approximately 0.92.

### Unique to Approach A
- **indicator**
  - Although the correlation between the above-predicted real estate transaction price and the label is 0.92, after actually plotting, it was found that the higher the predicted unit price of real estate transaction price, the lower the correlation with the label. Therefore, we added an indicator, which is marked as 1 when the predicted real estate transaction price is greater than 4.5, and marked as 0 for the rest, and manually marked the outlier in the upper left corner as 1.
    ![output](ryQnf-3Ba.png)
### Unique to Approach B
- Perform z-score normalization on all numerical features.
- Numerical features: [Building Age, Horizontal Coordinate, Vertical Coordinate, Predicted Real Estate Transaction Price, Any distance-related feature]

## Model settings
In this competition, we adopted the AutoML suite AutoGluon.
Since the evaluation metric of the competition is MAPE, after calculation, it is found that using MAE as the loss function + taking log() of y can best minimize MAPE, and the experimental results of doing so are also the best.

### Approach A
- Model: `autogluon.tabular.TabularPredictor`
- Hyperparameters:
  - `presets: best_quality`
  - base models: `[NN_TORCH, GBM, GBM_XT, GBM_Large, CAT, XGB, FASTAI_NN, RF, XT, KNN_uniform, KNN_normal]`
  - `eval_metric: mean_absolute_error`
  - hyperparameters: Changed the loss function of most base models from MSE to MAE.
- Target: `log(y)`

### Approach B
- Model: `autogluon.tabular.TabularPredictor`
- Hyperparameters:
  - `presets: best_quality`
  - base models: `[NN_TORCH, GBM, GBM_XT, GBM_Large, CAT, XGB, FASTAI_NN, RF, XT, KNN_uniform, KNN_normal]`
  - `eval_metric: mean_absolute_error`
  - hyperparameters: No special modifications, using default settings.
- Target: `log(y)`

## Training method
### Approach A
Mainly divided into two parts, first:

First, do distill + pseudo label + distill, and then do cluster correction after distill is done. The detailed methods are introduced separately below.

- **distill + pseudo label + distill**
  - **Method**
    - [STEP 1]: First, train a model A on the training data after the above feature engineering.
    - [STEP 2]: Distill model A into model A_dstl.
    - [STEP 3]: Use the predicted values of model A_dstl on public and private data as pseudo labels, and use training data (real label, only using original features, without adding feature engineering features) + public data (pseudo label) + private data (pseudo label) as training data to train a model B.
    - [STEP 4]: Distill model B into model B_dstl and make the final prediction.
  - **Explanation**
    - The models used above are all `autogluon.tabular.TabularPredictor`.
    - Distill uses the distill function implemented in `autogluon` itself, and sets `augment_data` to public data + private data.
    - `autogluon` also implements a pseudo label function, but we did not use it because we want to only use the original features to train the pseudo label.

- **cluster correction**
  - **Purpose**
    - Fine-tune the output of model B_dstl for fields that may be biased.
  - **Method**
    - [STEP 1]: Use training data, and select one field at a time to perform groupby.
    - [STEP 2]: Perform a linear regression on the grouped df, with the input being the prediction of model B_dstl and the target being the real label. Use the fitted model to adjust the predictions of public/private data.
    - [STEP 3]: Loop until all selected fields have been adjusted.
    - The order of fields used = [Building Age_quantile, City, Building Type, Total Floors_quantile, Transfer Level_quantile, Building Age_quantile, City, Main Use, Main Construction Material, Number of Parking Spaces_quantile = str(Field Value // 5)

### Approach B
- **city combination stacking**
  - **Method**
    - Use features after feature engineering.
    - [STEP 1]: Combine data from any two cities in the training data to train a first-stage model (a total of 153 first-stage models).
    - [STEP 2]: Add the predicted values of 153 first-stage models as additional features to the features after feature engineering, and then train a second-stage model and make the final prediction.

## Final Submission
The final submission result is Approach A * 0.7 + Approach B * 0.3