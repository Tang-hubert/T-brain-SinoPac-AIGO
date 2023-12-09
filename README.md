# T-brain-SinoPac-AIGO
## Done
- Have to fix the `y contains previously unseen labels` error.

## TO-DO
- Remember if you seperate the predict funtion, when facing different unknown labels the LabelEncoding will change. -> Have to code it in the same file.(main)
- lots of external_data try to use `Longitude` and `Latitude` and calculate each different installations and put (0,1) to recognized is it nearby or not.(Some are -1, have to ignore.)
- exacute some analysis from `pairplot_output.png`
    - Building_Area: against different Cities.
    - Age_of_Building: have to times -1
    - Parking_Area: Most are 0, and if it have wont effect the result(0,1)
    - Auxiliary_Building_Area: Most are 0, and if it have wont effect the result(0,1)
- Knowing how to evaluate the ML result.


# Good performance example: 永豐 Turing team 作法分享
> From [Sheng](https://hackmd.io/@shengchichi23/AI-GO-Turing-team)
## 主要分為 做法 A 與 做法 B ，最後將兩個做法的 prediction 做 weighted sum
以下分別介紹兩種做法在 feature engineering、model settings 以及 training method 的差異

## Feature engineering
本次比賽我們主要專注在收取外部資料以及使用實價登陸資料
以下介紹做法 A、B 共同以及不同的部分

### 做法 A、B 共同部分
- 將各資料的經緯度(WGS84)轉換成 TWD97 座標
- 將「金融機構基本資料.csv」透過欄位「金融機構名稱」分割成「銀行.csv」、「農會.csv」和「合作社.csv」
- 加入多個額外下載的外部資料，包含：
  - 變電所
  - 夜市
  - 公園
  - 發電廠
  - 資源回收公司
  - 加油站
  - 台北市公墓
  - 殯葬設施
  - 圖書館
  - 垃圾掩埋場
  - 寺廟
  - 汙水處理廠
  - 火車站點
  - 焚化廠
  - 警察局
  - 郵局
- 以 Manhattan distance 計算
  - 各個房屋與各個外部資料的最短距離
  - 1000 公尺內，是否有捷運站可提供假日的腳踏車租借服務
  - 1000 公尺內，最好的火車站級別
- 以 Euclidean distance 計算
  - 在 X 公尺內，各個外部資料的個數；X 由 hyperopt 搜尋，metric 為使得計算出來的個數與 label 的 correlation 絕對值越大越好
- 使用 2021~2022 年實價登錄資料，預測比賽資料的實價登錄價格，作為額外的一欄特徵
  - 訓練時使用到與比賽資料重和的欄位
  - 將類別型欄位盡可能 map 到與比賽資料同樣的類別
  - 將數值型欄位進行標準化，使得平均和標準差與比賽資料一致
  - 藉由 autogluon 進行訓練，使用到的參數如下
    - `time_limit=86400`
    - `presets="best_quality"`
  - 預測的實價登錄價格，在訓練資料上與 label 的 correlation 約莫 0.92

### 做法 A 特有
- **indicator**
  - 上述實價登錄預測價格與 label 的 correlation 雖然有 0.92，但實際畫圖後發現，若實價登錄預測單價越高，與 label 的相關性就越低，所以我們新增一個 indicator，當實價登錄預測價格大於 4.5 則標示為 1，其餘標示為 0，並且手動將左上角的 outlier 也標示為 1
    ![output](ryQnf-3Ba.png)
### 做法 B 特有
- 將所有數值型 feature 做 z-score normalization
- 數值型 feature: [屋齡, 橫坐標, 縱坐標, 實價登錄預測價格, 任何與距離相關之 feature]

## Model settings
本次比賽我們採用 AutoGluon 這個 AutoML 套件。
由於比賽的 evaluation metric 為 MAPE，經過計算後得知使用 MAE 作為 loss function + 對 y 取 log() 最能 minimize MAPE，而這樣做的實驗結果也最好。

### 做法 A
- Model: `autogluon.tabular.TabularPredictor`
- Hyperparameters:
  - `presets: best_quality`
  - base models: `[NN_TORCH, GBM, GBM_XT, GBM_Large, CAT, XGB, FASTAI_NN, RF, XT, KNN_uniform, KNN_normal]`
  - `eval_metric: mean_absolute_error`
  - hyperparameters: 將大部分 base model 的 loss function 從 MSE 改為 MAE
- Target: `log(y)`

### 做法 B
- Model: `autogluon.tabular.TabularPredictor`
- Hyperparameters:
  - `presets: best_quality`
  - base models: `[NN_TORCH, GBM, GBM_XT, GBM_Large, CAT, XGB, FASTAI_NN, RF, XT, KNN_uniform, KNN_normal]`
  - `eval_metric: mean_absolute_error`
  - hyperparameters: 沒有做特別修改，使用 default
- Target: `log(y)`

## Training method
### 做法 A
主要分為兩部分，首先

先做 distill + pseuodo label + distill 做完後再做 cluster correction，以下分別介紹其詳細作法

- **distill + pseuodo label + distill**
  - **做法**
    - [STEP 1]: 首先在經過上述 feature engineering 後的 training data 上訓練一個 model A
    - [STEP 2]: 將 model A distill 成 model A_dstl
    - [STEP 3]: 將 model A_dstl 在 public 跟 private 上的預測值作為 pseudo label，並使用 training data(real label，只使用原始 feature，沒有加入 feature engineering 的 feature) + public data(pseudo label) + private data(pseudo label) 作為訓練資料去訓練一個 model B
    - [STEP 4]: 將 model B distill 成 model B_dstl，並做最終預測
  - **說明**
    - 以上使用的模型皆為 `autogluon.tabular.TabularPredictor`
    - distill 是使用 `autogluon` 本身就有實作的 distill function，並將 `augment_data` 設定為 public data + private data
    - `autogluon` 也有實作 pseudo label function，但我們並沒有使用，因為我們希望只使用原始 feature 去訓練 pseudo label

- **cluster correction**
  - **目的**
    - 針對可能有 bias 的欄位, 對 model B_dstl 的 output 進行微調
  - **做法**
    - [STEP 1]: 使用 training data，每次挑選一個欄位進行 groupby
    - [STEP 2]: 對 grouped df 做一個 linear regression，input 是 model B_dstl 的 prediction，target 是 real label，用 fit 完的 model 去調整 public / private 的 prediction
    - [STEP 3]: for 迴圈直到所有挑選的欄位都被調整過
    - 使用的欄位順序 = [屋齡_quantile, 縣市, 建物型態, 總樓層數_quantile, 移轉層次_quantile, 屋齡_quantile, 縣市, 主要用途, 主要建材, 車位個數
      _quantile = str(欄位數值 // 5)

### 做法 B
- **city combination stacking**
  - **做法**
    - 使用經過 feature engineering 後的 feature
    - [STEP 1]: 組合 training data 中任兩縣市的資料去訓練一階模型(共 153 個一階模型)
    - [STEP 2]: 將經過 feature engineering 後的 feature 加入 153 個一階模型的預測值當作額外 feature，再訓練一個二階模型並做最終預測

## 最終提交
最終提交之結果為 做法 A * 0.7 + 做法 B * 0.3