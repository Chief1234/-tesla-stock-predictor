# Tesla Stock Price Prediction using LSTM and RNN

This project predicts Tesla stock prices using deep learning models built in **Google Colab** with **LSTM** and **SimpleRNN** networks. The notebook covers data preprocessing, exploratory data analysis, sequence generation, hyperparameter tuning, model training, forecasting, evaluation, and deployment file preparation.[file:25]

## Overview

The project uses historical Tesla stock market data from `TSLA.csv` with features such as **Open, High, Low, Close, Adj Close, and Volume**.[file:25] The notebook converts the `Date` column into a datetime index, applies missing-value handling, scales the data with `MinMaxScaler`, and trains recurrent neural networks for stock prediction.[file:25]

The notebook compares **LSTM** and **SimpleRNN** models across multiple forecasting horizons of **1 day, 5 days, and 10 days**.[file:25] It also saves trained model files, a scaler file, and deployment-related files for later use.[file:25]

## Features

- Tesla stock data preprocessing and cleaning.[file:25]
- Exploratory data analysis using line plots and correlation heatmaps.[file:25]
- Feature scaling with `MinMaxScaler`.[file:25]
- Sequence generation for multi-horizon forecasting.[file:25]
- LSTM and SimpleRNN model training.[file:25]
- Hyperparameter tuning for units and dropout.[file:25]
- Model comparison using MSE.[file:25]
- Export of trained models and scaler for deployment.[file:25]

## Dataset

The notebook reads Tesla historical stock data from `TSLA.csv` and uses these columns:[file:25]

- Open
- High
- Low
- Close
- Adj Close
- Volume

After preprocessing, the notebook reports a cleaned dataset shape of **2416 rows and 6 columns**.[file:25]

## Tech Stack

- Python
- Google Colab
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Joblib
- Streamlit (deployment prep in notebook)[file:25]

## Project Workflow

### 1. Data Loading
The notebook loads `TSLA.csv`, parses the `Date` column, and sets it as the dataframe index for time-series analysis.[file:25]

### 2. Data Cleaning
Missing values are handled using forward fill and backward fill before final cleanup.[file:25]

### 3. Exploratory Data Analysis
The notebook creates visualizations for:
- Tesla adjusted close trend.[file:25]
- Recent trading volume.[file:25]
- Open vs Close price comparison.[file:25]
- Feature correlation heatmap.[file:25]

### 4. Feature Scaling
The project scales stock features using `MinMaxScaler` before model training, and the scaler is saved as `scaler.pkl`.[file:25]

### 5. Sequence Generation
A custom `create_sequences()` function is used to generate input-output samples for supervised learning.[file:25] The notebook builds sequences for **1-day**, **5-day**, and **10-day** prediction horizons.[file:25]

### 6. Model Building
Two deep learning models are created:
- **LSTM model** using stacked LSTM layers, dropout, and dense layers.[file:25]
- **SimpleRNN model** using stacked recurrent layers, dropout, and dense layers.[file:25]

### 7. Hyperparameter Tuning
The notebook tests multiple combinations of:
- Units: **50**, **100**
- Dropout: **0.2**, **0.3**[file:25]

The best parameters reported in the notebook are:
- **Best LSTM:** 50 units, 0.3 dropout.[file:25]
- **Best RNN:** 100 units, 0.3 dropout.[file:25]

### 8. Training and Evaluation
Both models are trained using `EarlyStopping` and `ModelCheckpoint` callbacks for each horizon.[file:25] The project evaluates predictions using **Mean Squared Error (MSE)** after inverse scaling.[file:25]

## Final Results

The notebook prints the following final MSE comparison:[file:25]

| Horizon | LSTM MSE | RNN MSE |
|--------|---------:|--------:|
| 1 Day | 9.744017e+11 [file:25] | 2.116687e+12 [file:25] |
| 5 Days | 3.089312e+12 [file:25] | 2.534712e+12 [file:25] |
| 10 Days | 2.841433e+12 [file:25] | 6.119007e+12 [file:25] |

## Files Generated

The notebook saves or prepares these files:[file:25]

- `scaler.pkl`
- `lstmh1.h5`
- `lstmh5.h5`
- `lstmh10.h5`
- `rnnh1.h5`
- `rnnh5.h5`
- `rnnh10.h5`
- `requirements.txt`
- `tesla_streamlit_app.zip` or similar deployment zip output.[file:25]

## Run in Google Colab

1. Open **Google Colab**.
2. Upload `Tesla_Stock_Price_Prediction_Project.ipynb`.
3. Upload `TSLA.csv` into the Colab session.
4. Run all notebook cells in order.
5. Download generated model and scaler files after training.[file:25]

## Installation

If you want to run the project manually, install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib plotly yfinance streamlit
```

The notebook also creates a `requirements.txt` file for deployment preparation.[file:25]

## Note

The notebook includes an install command for `tensorflow==2.15.0`, but the captured output shows a version resolution issue in Colab before TensorFlow 2.19.0 is shown in the environment.[file:25] If that happens again, update the TensorFlow version in Colab to a compatible one.[file:25]

## Future Improvements

- Add RMSE, MAE, and R-squared metrics.
- Try GRU and Transformer-based architectures.
- Add live stock data integration.
- Improve deployment with a complete Streamlit app.
- Use sentiment or macroeconomic indicators as additional features.

## Author

Harshit Chandra (Aspiring Data Science and Analytics) 
Data Science / Machine Learning Project

## License

This project is for educational and portfolio use.
