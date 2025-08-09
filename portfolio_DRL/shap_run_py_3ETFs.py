import sys

from SVERL_icml_2023.portfolio_DRL.data_function import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from SVERL_icml_2023.portfolio_DRL.create_model import *
from sklearn.model_selection import TimeSeriesSplit
from SVERL_icml_2023.shapley import Shapley
import shap
import time
sys.path.append("E:/XAI/RL_with_SHAP/pythonProject1/SVERL_icml_2023")
# from portfolio_DRL.create_model import StockPredictWrapper
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import pandas as pd
import numpy as np
import gymnasium as gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

# Example usage
input_dir = "E:\XAI\RL_with_SHAP\pythonProject1\SVERL_icml_2023\portfolio_DRL"  # Use the specified directory
ticker_file = os.path.join(input_dir, "tickers.txt")  # Text file containing tickers, one per line
factor_file = os.path.join(input_dir, "FF_factors.csv")  # CSV file containing factor returns
start_date = "2005-01-01"
end_date = "2024-12-31"
output_csv = os.path.join(input_dir, "stock_prices.csv")

factors = pd.read_csv(f"{input_dir}/aligned_factors.csv", index_col=0, parse_dates=True)
returns_raw = pd.read_csv(f"{input_dir}/daily_returns_DIA.csv", index_col=0, parse_dates=True)
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Remove NaN values and align dates
factors = factors.dropna()
returns = returns_raw.dropna()

# Align dates between factors and returns
dates = factors.index.intersection(returns.index)
factors_raw = factors.loc[dates]
returns_raw = returns.loc[dates]
factors_raw = factors_raw.iloc[:, :-1]
# print(f"factors_raw = {factors_raw}")

Nday_diff = 1  # Define the future prediction horizon for shift days
# limit to 2 stock ONLY
returns_raw = returns_raw.iloc[:, :1]

# Remove NaN values and align dates
factors_raw = factors_raw.dropna()
returns_raw = returns_raw.dropna()
# factors_raw = pd.concat([factors_raw, returns_raw], axis=1)
# Align dates between factors_raw and returns_raw
dates = factors_raw.index.intersection(returns_raw.index)
factors_raw = factors_raw.loc[dates]
returns_raw = returns_raw.loc[dates]

# Shift returns to generate target variable (N-day future return)
returns_raw_Y = returns_raw.shift(-Nday_diff).dropna()

# Align all datasets with available dates in returns_raw_Y
dates_final = factors_raw.index.intersection(returns_raw_Y.index)
factors_raw = factors_raw.loc[dates_final]
returns_raw = returns_raw.loc[dates_final]
returns_raw_Y = returns_raw_Y.loc[dates_final]

N = 1
sequence_length = 10
batch_size = 64
num_epochs = 500
learning_rate = 0.001
rolling_window_train = 360  # Reduce for faster adaptation
rolling_window_test = 10  # Reduce test window
prediction_window = 10  # Align prediction period with test window
train_model2 = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Record results
results = []
prediction_records = []
shapley_records = []
shapley_metrics_df = []
all_test_losses = []
all_binary_test_losses = []
results = []
classification_results = []
shap_metrics_results = []
test_results = []
prediction_results = []
combined_test_result = []
portfolio_records = []
start = 0  # Initialize rolling split index

# Rolling split: Train (M days), Test (N days), Predict (W days)
total_samples = len(factors_raw)
# total_samples = 382
print(f'total_samples = {total_samples}')
print(f'total sum = {rolling_window_train + rolling_window_test + prediction_window}')

while start + rolling_window_train + rolling_window_test + prediction_window <= total_samples:
    start_time = time.time()
    train_start = start
    train_end = start + rolling_window_train

    test_start = train_end
    test_end = test_start + rolling_window_test

    predict_start = test_end  # Predictions start right after testing
    predict_end = predict_start + prediction_window

    # Extract the actual date indices
    train_dates = (factors_raw.iloc[train_start].name, factors_raw.iloc[train_end - 1].name)
    test_dates = (factors_raw.iloc[test_start].name, factors_raw.iloc[test_end - 1].name)
    predict_dates = (factors_raw.iloc[predict_start].name, factors_raw.iloc[predict_end - 1].name)

    print(f"\n🔹 Processing Date Window: Train {train_dates}, Test {test_dates}, Predict {predict_dates}")
    print(f'returns_raw.columns = {returns_raw.columns}')
    for stock in returns_raw.columns:
        print(f"Training for stock: {stock}")
        stock_returns = returns_raw_Y[[stock]]  # Select the stock column
        # stock_returns = returns_raw[[stock]].shift(-1).dropna()
        # stock_returns_mean = returns_raw_Y[[stock]].rolling(window=N, min_periods=N).mean().shift(-N+1) # calc average over next N period
        # stock_returns = stock_returns_mean.dropna()

        stock_returns_unshift = returns_raw[[stock]]

        factors_raw_X = pd.concat([factors_raw, stock_returns_unshift], axis=1)
        # Rolling split: Train (M days), Test (N days), Predict (W days)
        # total_samples = len(factors_raw_X)
        # Initialize the rolling split index

        # Print the selected date ranges
        print(f"Training dates: {train_dates}")
        print(f"Testing dates: {test_dates}")
        print(f"Prediction dates: {predict_dates}")

        # # Split data into training, testing, and prediction sets
        train_factors = factors_raw_X.iloc[train_start:train_end]
        train_returns = stock_returns.iloc[train_start:train_end]
        test_factors = factors_raw_X.iloc[test_start:test_end]
        test_returns = stock_returns.iloc[test_start:test_end]
        predict_factors = factors_raw_X.iloc[predict_start:predict_end]
        predict_returns = stock_returns.iloc[predict_start:predict_end]
        #
        # # Normalize train and test data
        train_factors_mean, train_factors_std = train_factors.mean(), train_factors.std()
        train_returns_mean, train_returns_std = train_returns.mean(), train_returns.std()
        # print(f'train_factors_mean ={train_factors_mean}')
        # Avoid division by zero
        train_factors_std.replace(0, 1e-8, inplace=True)
        train_returns_std.replace(0, 1e-8, inplace=True)

        train_factors_norm = (train_factors - train_factors_mean) / train_factors_std
        train_returns_norm = (train_returns - train_returns_mean) / train_returns_std
        test_factors_norm = (test_factors - train_factors_mean) / train_factors_std
        test_returns_norm = (test_returns - train_returns_mean) / train_returns_std
        predict_factors_norm = (predict_factors - train_factors_mean) / train_factors_std
        predict_returns_norm = (predict_returns - train_returns_mean) / train_returns_std
        # print(f'train_factors_norm dim: {train_factors_norm.shape}')
        # print(f'train_returns_norm dim: {train_returns_norm.shape}')
        dataset = StockDataset(train_factors_norm, train_returns_norm, sequence_length, N=1)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # for batch_x, batch_y in dataloader:
        #     print(f"Batch X Shape: {batch_x.shape}")  # Expected: (batch_size, sequence_length, num_features)
        #     print(f"Batch Y Shape: {batch_y.shape}")  # Expected: (batch_size, N, num_features)
        #     break  # Only print for the first batch
        # # Model initialization
        d_model = train_factors.shape[1]
        # print(f'd_model: {d_model}')

        num_heads = min(4,
                        d_model)  # Ensure num_heads does not exceed d_model and d_model is divisible by num_heads
        while d_model % num_heads != 0:
            num_heads -= 1
        # print(f"d_model: {d_model} num_heads: {num_heads} sequence_length: {sequence_length} and train factor return dim: {train_factors.shape[1]} {train_returns.shape[1]}")
        model = StockPredictAgent(d_model=d_model, num_heads=num_heads, sequence_length=sequence_length, N=1).to(
            device)
        #
        # # Optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        criterion = nn.SmoothL1Loss(beta=1.0)  # Huber Loss for stability
        #
        # # Training the model
        # print(f"Training window: {train_start} to {train_end}")
        # print(f"Expected FC Layer Input Shape: {sequence_length * d_model}")
        # print(f"Flattened Tensor Shape: {batch_size, sequence_length * d_model}")

        train_model_with_logging(model, dataloader, optimizer, criterion, scheduler, num_epochs, device)

        print(f'Finished training model 1')
        # Create directory to save models


        # Format filename with stock and training end date
        model_filename = f"{save_dir}/model_{stock}_{train_dates[1].strftime('%Y%m%d')}.pt"

        # Save model state dict
        torch.save(model.state_dict(), model_filename)
        print(f"✅ Model saved: {model_filename}")

        # print(f'train_factors_norm dim: {train_factors_norm.shape}')
        # Convert to PyTorch tensors
        # print(f'test_factors_tensor: {test_factors_tensor.shape}')
        # Run inference on test data


        augmented_test_features, shap_metrics_test = generate_augmented_features_testphase(
            test_factors_norm, train_factors_norm, train_factors_mean, train_factors_std,
            model, prediction_window, device, return_metrics=True
        )
        shap_metrics_test["Stock"] = stock
        shap_metrics_test["Start_Date"] = test_dates[0]
        shap_metrics_test["End_Date"] = test_dates[1]
        shap_metrics_test["Phase"] = 'Test'
        shap_metrics_results.append(shap_metrics_test.iloc[0].tolist())

    # start += prediction_window
    start += 1
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")


shap_metrics_results_df = pd.DataFrame(shap_metrics_results, columns=[
    "mean_abs_shap_Mkt-RF", "mean_abs_shap_SMB", "mean_abs_shap_HML", "mean_abs_shap_RMW", "mean_abs_shap_CMA",
    "mean_abs_shap_Stock",
    "mean_shap_Mkt-RF", "mean_shap_SMB", "mean_shap_HML", "mean_shap_RMW", "mean_shap_CMA", "mean_shap_Stock",
    "shap_std_Mkt-RF", "shap_std_SMB", "shap_std_HML", "shap_std_RMW", "shap_std_CMA", "shap_std_Stock",
    "mean_over_std_Mkt-RF", "mean_over_std_SMB", "mean_over_std_HML", "mean_over_std_RMW", "mean_over_std_CMA",
    "mean_over_std_Stock",
    "mean_abs_over_std_Mkt-RF", "mean_abs_over_std_SMB", "mean_abs_over_std_HML", "mean_abs_over_std_RMW",
    "mean_abs_over_std_CMA", "mean_abs_over_std_Stock",
    "feature_importance_ranking_Mkt-RF", "feature_importance_ranking_SMB", "feature_importance_ranking_HML",
    "feature_importance_ranking_RMW", "feature_importance_ranking_CMA", "feature_importance_ranking_Stock",
    "Stock", "Start_Date", "End_Date", "Phase"
])

shap_metrics_results_df.to_csv("shap_value_metrics_export_DIA.csv", index=False)
print("shap_value_metrics_export saved successfully!")

import pandas as pd
import numpy as np
import ta

# Load DIA price data (replace with your actual path)
etf = pd.read_csv("stock_prices_DIA.csv", parse_dates=["Date"])
etf.set_index("Date", inplace=True)

dia = etf[[stock]].copy()  # ✅ returns DataFrame
dia.rename(columns={stock: "Close"}, inplace=True)  # Rename to 'Close'

# Compute indicators
dia["Return_1d"] = dia["Close"].pct_change()
dia["Return_5d"] = dia["Close"].pct_change(5)
dia["Rolling_Return_21d"] = dia["Close"].pct_change(21)
dia["Volatility_5d"] = dia["Return_1d"].rolling(window=5).std()
dia["Volatility_21d"] = dia["Return_1d"].rolling(window=21).std()
dia["Drawdown"] = dia["Close"] / dia["Close"].cummax() - 1

dia["SMA_20"] = ta.trend.sma_indicator(dia["Close"], window=20)
dia["SMA_50"] = ta.trend.sma_indicator(dia["Close"], window=50)
dia["EMA_12"] = ta.trend.ema_indicator(dia["Close"], window=12)
dia["EMA_26"] = ta.trend.ema_indicator(dia["Close"], window=26)
dia["MACD"] = ta.trend.macd_diff(dia["Close"])
dia["RSI_14"] = ta.momentum.rsi(dia["Close"], window=14)

bb = ta.volatility.BollingerBands(close=dia["Close"], window=20)
dia["BB_high"] = bb.bollinger_hband()
dia["BB_low"] = bb.bollinger_lband()
atr = ta.volatility.AverageTrueRange(high=dia["Close"] * 1.01,
                                     low=dia["Close"] * 0.99,
                                     close=dia["Close"])
dia["ATR_14"] = atr.average_true_range()

# Reset index
dia.reset_index(inplace=True)

# Load SHAP metrics
shap_df = pd.read_csv("shap_value_metrics_export_DIA.csv")
shap_df["End_Date"] = pd.to_datetime(shap_df["End_Date"])

# Merge
merged = pd.merge(shap_df, dia, how="left", left_on="End_Date", right_on="Date")
merged.drop(columns=["Date"], inplace=True)
merged.to_csv("shap_with_DIA_indicators.csv", index=False)
print("✅ Saved: shap_with_DIA_indicators.csv")