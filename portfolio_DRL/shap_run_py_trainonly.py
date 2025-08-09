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


# try:
#     tickers = load_tickers_from_file(ticker_file)
#     price_data = download_stock_prices(tickers, start_date, end_date, output_csv)
#     daily_returns = calculate_daily_returns(price_data, return_type='arithmetic')
#     daily_returns.to_csv(os.path.join(input_dir, "daily_returns.csv"))
#     factor_returns = read_factor_returns(factor_file, daily_returns)
#     factor_returns.to_csv(os.path.join(input_dir, "aligned_factors.csv"))
#     print("Factor returns alignment completed.")
# except Exception as e:
#     print(f"An error occurred: {e}")

factors = pd.read_csv(f"{input_dir}/aligned_factors.csv", index_col=0, parse_dates=True)
returns_raw = pd.read_csv(f"{input_dir}/daily_returns.csv", index_col=0, parse_dates=True)


# Remove NaN values and align dates
factors = factors.dropna()
returns = returns_raw.dropna()

# Align dates between factors and returns
dates = factors.index.intersection(returns.index)
factors_raw = factors.loc[dates]
returns_raw = returns.loc[dates]
factors_raw = factors_raw.iloc[:, :-1]
# print(f"factors_raw = {factors_raw}")

Nday_diff = 10  # Define the future prediction horizon for shift days
# limit to 2 stock ONLY
# returns_raw = returns_raw.iloc[:, :2]

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

N = 10
sequence_length = 10
batch_size = 64
num_epochs = 500
learning_rate = 0.001
rolling_window_train = 252  # Reduce for faster adaptation
rolling_window_test = 10    # Reduce test window
prediction_window = 10      # Align prediction period with test window
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

# state = 'train_data_generate'
state = 'train_data_generate'
# state = 'train_RL'
# state = 'SHAP_optimal_weight'

if state == 'train_data_generate':
    # Rolling split: Train (M days), Test (N days), Predict (W days)
    # total_samples = len(factors_raw)
    total_samples = 382
    print(f'total_samples = {total_samples}')
    print(f'total sum = {rolling_window_train + rolling_window_test + prediction_window}')
    while start + rolling_window_train + rolling_window_test + prediction_window <= total_samples:

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

        for stock in returns_raw.columns:
            print(f"Training for stock: {stock}")
            stock_returns = returns_raw_Y[[stock]]  # Select the stock column
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
            dataset = StockDataset(train_factors_norm, train_returns_norm, sequence_length, N=N)

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
            model = StockPredictAgent(d_model=d_model, num_heads=num_heads, sequence_length=sequence_length, N=N).to(
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
            # print(f'train_factors_norm dim: {train_factors_norm.shape}')
            # Convert to PyTorch tensors
            # print(f'test_factors_tensor: {test_factors_tensor.shape}')
            # Run inference on test data
            if train_model2 == True:
                augmented_train_features, shap_metrics = generate_augmented_features(
                    train_factors_norm, train_factors_mean, train_factors_std, model, rolling_window_test, device, return_metrics=True
                )
                augmented_features_tensor = torch.tensor(augmented_train_features, dtype=torch.float32).to(device)

                print(
                    f"Final Augmented Tensor Shape: {augmented_train_features.shape}")  # Expected: (rolling_window_test, d_model * 2)

                # Convert test_returns_norm into binary classification
                mean_return_binary = np.mean(train_returns_norm)  # Aggregate 60-day return
                train_returns_threshold = (0 - train_returns_mean) / train_returns_std  # Convert zero into normalized scale
                train_returns_binary = (mean_return_binary > train_returns_threshold).astype(
                    int)  # 1 if positive, 0 if negative
                # print(f'train_returns_norm = {train_returns_norm}')
                print(f"Cumulative Return over 60 days: {mean_return_binary}")
                print(f"Binary Target (Up=1, Down=0): {train_returns_binary}")

            # Train the Model Using Augmented Features
            # sequence_length = 20
            # batch_size = 32
            # num_epochs = 25
            # learning_rate = 0.001
                d_model2 = augmented_train_features.shape[1]  # Number of features (6 original + 6 SHAP)
                num_heads = min(4, d_model2)
                while d_model2 % num_heads != 0:
                    num_heads -= 1

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Length of augmented_test_features: {augmented_train_features.shape}")
                print(f"Length of sequence: {sequence_length}")

                # sequence_length2 = min(10, len(augmented_test_features))  # Ensure it's smaller than available data
                sequence_length2 = prediction_window
                # Create dataset and dataloader for training
                dataset2 = StockDatasetBinary(augmented_train_features, train_returns_norm, sequence_length2)
                dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

                for batch_x, batch_y in dataloader2:
                    print(f"Batch X Shape: {batch_x.shape}")  # Expected: (batch_size, sequence_length, num_features)
                    print(f"Batch Y Shape: {batch_y.shape}")  # Expected: (batch_size, N, num_features)
                    break  # Only print for the first batch

                # Initialize the binary classification model
                model_binary = StockPredictAgentBinary(d_model=d_model2, num_heads=num_heads,
                                                       sequence_length=sequence_length2).to(device)

                # Define optimizer and loss function
                optimizer = optim.Adam(model_binary.parameters(), lr=learning_rate, weight_decay=1e-5)
                criterion = nn.BCELoss()  # Binary cross-entropy loss

                # Train the model
                train_model(model_binary, dataloader2, optimizer, criterion, num_epochs, device)
                print(f'Finished training model 2')

                test_factors_tensor = torch.tensor(test_factors_norm.values, dtype=torch.float32).to(device).unsqueeze(0)
                predict_factors_tensor = torch.tensor(predict_factors_norm.values, dtype=torch.float32).to(
                    device).unsqueeze(0)

                model.eval()
                with torch.no_grad():
                    test_predictions_norm = model(test_factors_tensor).cpu().numpy().flatten()
                    future_predictions_norm = model(predict_factors_tensor).cpu().numpy().flatten()

                    # test_output = model(torch.tensor(test_factors_np, dtype=torch.float32).to(device))
                    # print(f"Model output shape: {test_output.shape}")  # Expected: (1, 20) for 20 future returns
                # print(f'test_factors_tensor: {test_factors_tensor.shape}')
                # Revert normalization for test and predicted returns
                test_predictions = test_predictions_norm * train_returns_std.item() + train_returns_mean.item()
                future_predictions = future_predictions_norm * train_returns_std.item() + train_returns_mean.item()

                augmented_test_features = generate_augmented_features_testphase(
                    test_factors_norm, train_factors_norm, train_factors_mean, train_factors_std,
                    model, prediction_window, device
                )
                # ✅ Convert NumPy array to PyTorch tensor
                augmented_features_tensor_test = torch.tensor(augmented_test_features, dtype=torch.float32).to(device)

                # ✅ Ensure the tensor is 3D by adding a batch dimension at index 0
                augmented_features_tensor_test = augmented_features_tensor_test.unsqueeze(0)  # Shape becomes (1, 20, 12)

                augmented_predict_features = generate_augmented_features_testphase(
                    predict_factors_norm, train_factors_norm, train_factors_mean, train_factors_std,
                    model, prediction_window, device
                )
                # ✅ Convert NumPy array to PyTorch tensor
                augmented_features_tensor_test = torch.tensor(augmented_test_features, dtype=torch.float32).to(device)
                augmented_features_tensor_predict = torch.tensor(augmented_predict_features, dtype=torch.float32).to(device)

                # ✅ Ensure the tensor is 3D by adding a batch dimension at index 0
                augmented_features_tensor_test = augmented_features_tensor_test.unsqueeze(0)  # Shape becomes (1, 20, 12)
                augmented_features_tensor_predict = augmented_features_tensor_predict.unsqueeze(
                    0)  # Shape becomes (1, 20, 12)

                # ✅ Debugging: Print final shape
                print(
                    f'✅ Transformed augmented_features_tensor_test Shape: {augmented_features_tensor_test.shape}')  # Expected: (1, 20, 12)

                # Get model predictions
                model_binary.eval()
                with torch.no_grad():
                    return_test_prob = model_binary(augmented_features_tensor_test).cpu().numpy().flatten()
                    return_predict_prob = model_binary(augmented_features_tensor_predict).cpu().numpy().flatten()

                print(f'fnished 2nd model test')
                # Convert probabilities to binary labels (0 = down, 1 = up)
                # Convert probability outputs to binary labels (0 = down, 1 = up)
                return_test_binary = (return_test_prob > 0.5).astype(int).item()
                return_predict_binary = (return_predict_prob > 0.5).astype(int).item()

                # TO-DO: add test phase stats (loss from both models)
                # TO-DO: add prediction data from both models, find stocks that go ups and downs with their predicted return series

                # future_predictions = future_predictions_norm * train_returns_std.item() + train_returns_mean.item()
                actual_test_returns = test_returns.values.flatten()
                actual_predict_returns = predict_returns.values.flatten()
                actual_test_returns_binary = (np.mean(actual_test_returns) > 0).astype(int)
                actual_predict_returns_binary = (np.mean(actual_predict_returns) > 0).astype(int)

                # print(f'test_predictions dim:{test_predictions.shape} actual_test_returns dim: {actual_test_returns.shape}')

                # Compute loss for test and prediction phases
                test_loss = np.mean((test_predictions - actual_test_returns) ** 2)
                predict_loss = np.mean((future_predictions - actual_predict_returns) ** 2)

                binary_test_loss = np.mean(np.abs(return_test_binary - actual_test_returns_binary))
                binary_predict_loss = np.mean(np.abs(return_predict_binary - actual_predict_returns_binary))

                all_test_losses.append(test_loss)
                print(f'test_loss: {test_loss}')
                all_binary_test_losses.append(binary_test_loss)
                print(f'binary test_loss: {binary_test_loss}')

                # Store test results
                combined_test_result = {
                    "Stock": stock,
                    "Test Dates": (test_start, test_end),
                    "Predicted Returns (Test Set)": test_predictions.tolist(),
                    "Actual Returns (Test Set)": actual_test_returns.tolist(),
                    "Loss": test_loss
                }

                # Store time series results (date-wise)
                for i in range(len(test_predictions)):
                    results.append([
                        stock,
                        "test",
                        test_returns.index[i],  # Store the date
                        test_predictions[i],
                        actual_test_returns[i]
                    ])

                for i in range(len(future_predictions)):
                    results.append([
                        stock,
                        "predict",
                        predict_returns.index[i],  # Store the date
                        future_predictions[i],
                        actual_predict_returns[i]
                    ])

                # calculate 60 days vol for stock i
                stock_historical_returns_test = stock_returns_unshift.iloc[(train_end - 60):train_end]
                stock_historical_returns_predict = stock_returns_unshift.iloc[(test_end - 60):test_end]

                stock_volatility_test = stock_historical_returns_test.std().values[0]
                stock_volatility_predict = stock_historical_returns_predict.std().values[0]

                # Store scalar classification results (with start & end dates)
                classification_results.append([
                    stock, "test_binary", test_dates[0], test_dates[1], return_test_binary, return_test_prob,
                    actual_test_returns_binary, stock_volatility_test, float(np.prod(test_returns.values + 1) - 1)

                ])
                classification_results.append([
                    stock, "predict_binary", predict_dates[0], predict_dates[1], return_predict_binary, return_predict_prob,
                    actual_predict_returns_binary, stock_volatility_predict, float(np.prod(predict_returns.values + 1) - 1)
                ])
                print(f'return_predict_prob = {return_predict_prob}')
                print(f"Loss for {stock}: Test Loss: {test_loss:.6f}, Predict Loss: {predict_loss:.6f}")
                print(
                    f"Binary Model Loss for {stock}: Test Loss: {binary_test_loss:.6f}, Predict Loss: {binary_predict_loss:.6f}")

                # Convert results to DataFrame
                # Add SHAP feature importance metrics to the test results
                # combined_test_result.update(feature_importance_df_pivot.iloc[0].to_dict())

                # Append the combined results to the test_results list
                # test_results.append(combined_test_result)
                #
                # for date, predicted, actual in zip(predict_returns.index, future_predictions, actual_predict_returns):
                #     prediction_results.append({
                #         "Stock": stock,
                #         "Date": date,
                #         "Predicted Return": predicted,
                #         "Actual Return": actual
                #     })
                # print(f"Predictions for {stock} from {predict_start} to {predict_end} complete.")
                # Print average test loss for this stock
                avg_test_loss = np.mean(all_test_losses)
                avg_binary_test_loss = np.mean(all_binary_test_losses)
                print(
                    f"Average Test Loss for {stock}: {avg_test_loss:.6f} and Average Binary Test Loss is {avg_binary_test_loss:.6f}")
            elif train_model2 == False:
                augmented_train_features, shap_metrics_train = generate_augmented_features(
                    train_factors_norm, train_factors_mean, train_factors_std, model, rolling_window_test, device,
                    return_metrics=True
                )
                shap_metrics_train["Stock"] = stock
                shap_metrics_train["Start_Date"] = train_dates[0]
                shap_metrics_train["End_Date"] = train_dates[1]
                shap_metrics_train["Phase"] = 'Train'
                shap_metrics_results.append(shap_metrics_train.iloc[0].tolist())

                augmented_test_features, shap_metrics_test = generate_augmented_features_testphase(
                    test_factors_norm, train_factors_norm, train_factors_mean, train_factors_std,
                    model, prediction_window, device, return_metrics=True
                )
                shap_metrics_test["Stock"] = stock
                shap_metrics_test["Start_Date"] = test_dates[0]
                shap_metrics_test["End_Date"] = test_dates[1]
                shap_metrics_test["Phase"] = 'Test'
                shap_metrics_results.append(shap_metrics_test.iloc[0].tolist())

                augmented_predict_features, shap_metrics_predict = generate_augmented_features_testphase(
                    predict_factors_norm, train_factors_norm, train_factors_mean, train_factors_std,
                    model, prediction_window, device, return_metrics=True
                )
                shap_metrics_predict["Stock"] = stock
                shap_metrics_predict["Start_Date"] = predict_dates[0]
                shap_metrics_predict["End_Date"] = predict_dates[1]
                shap_metrics_predict["Phase"] = 'Predict'
                shap_metrics_results.append(shap_metrics_predict.iloc[0].tolist())
                print(f'---Append Shape value metrics for {stock} and last date is {predict_dates[1]}---')

        if train_model2 == True:
            # Convert classification_results into a DataFrame
            classification_df = pd.DataFrame(classification_results, columns=[
                "Stock", "Phase", "Start Date", "End Date", "Predicted Binary", "Predicted Probability",
                "Actual Binary", "Volatility", "Next Period Return"
            ])

            # 🔹 Filter for a specific test period (test_dates[1])
            selected_test_date = predict_dates[1]  # Replace with the target test end date
            filtered_df = classification_df[(classification_df["Phase"] == "predict_binary") &
                                            (classification_df["End Date"] == selected_test_date)]

            # 🔹 Extract required inputs
            stock_tickers = filtered_df["Stock"].tolist()
            predicted_probs = filtered_df["Predicted Probability"].values.astype(float)
            rolling_volatility = filtered_df["Volatility"].values.astype(float)
            print(f'predicted_probs={predicted_probs}')
            # # 🔹 Map stocks to sectors (Assume `sector_dict` contains stock-sector mapping)
            # sector_dict = {
            #     'BA': 'Industrials',
            #     'AMGN': 'Healthcare',
            #     'DIS': 'Communication Services',
            #     'NKE': 'Consumer Discretionary',
            #     'HON': 'Industrials',
            #     'MMM': 'Industrials',
            #     'CAT': 'Industrials',
            #     'KO': 'Consumer Staples',
            #     'PG': 'Consumer Staples',
            #     'AXP': 'Financials',
            #     'GS': 'Financials',
            #     'JPM': 'Financials',
            #     'MCD': 'Consumer Discretionary',
            #     'HD': 'Consumer Discretionary',
            #     'AAPL': 'Technology',
            #     'CRM': 'Technology',
            #     'CSCO': 'Technology',
            #     'IBM': 'Technology',
            #     'MSFT': 'Technology',
            #     'TRV': 'Financials',
            #     'UNH': 'Healthcare',
            #     'CVX': 'Energy',
            #     'JNJ': 'Healthcare',
            #     'MRK': 'Healthcare',
            #     'AMZN': 'Consumer Discretionary',
            #     'WMT': 'Consumer Staples',
            #     'INTC': 'Technology',
            #     'VZ': 'Communication Services'
            # }  # Example mapping, replace with real sector data
            # sectors = [sector_dict.get(stock, "Unknown") for stock in stock_tickers]  # Default to "Unknown" if missing

            # TO-DO: i can use test period data to fine tune confidence_threshold and sector_limit parameter and then trade on prediction period:
            # 🔹 Call compute_portfolio_weights function
        ###################################################################
        # UNCOMMENT FOR TRADING
        #     portfolio_weights = compute_portfolio_weights(
        #         stock_tickers=stock_tickers,
        #         predicted_probs=predicted_probs,
        #         rolling_volatility=rolling_volatility,
        #         sectors=sectors,
        #         confidence_threshold=(0.353392545,	0.662810753),  # Exclude low-confidence predictions
        #         short_ratio=0.262342442,  # Max 30% per sector
        #         annualize_volatility=False  # Convert daily volatility to annualized if needed
        #     )
        #     bmk_weights_abs_wt = compute_portfolio_weights_BMK(
        #         stock_tickers=stock_tickers,
        #         predicted_probs=predicted_probs,
        #         rolling_volatility=rolling_volatility,
        #         sectors=sectors,
        #         confidence_threshold=(0.0, 0.0),  # Exclude low-confidence predictions
        #         short_ratio=0,  # Max 30% per sector
        #         annualize_volatility=False,
        #         raw_weight_version =1
        #     )
        #     bmk_weights_raw_pred = compute_portfolio_weights_BMK(
        #         stock_tickers=stock_tickers,
        #         predicted_probs=predicted_probs,
        #         rolling_volatility=rolling_volatility,
        #         sectors=sectors,
        #         confidence_threshold=(0.0, 0.0),  # Exclude low-confidence predictions
        #         short_ratio=0,  # Max 30% per sector
        #         annualize_volatility=False,
        #         raw_weight_version=2
        #     )
        #     # 🔹 Display final portfolio allocation
        #     print("\n🔹 Final Portfolio Weights for Test Date:", selected_test_date)
        #     print(portfolio_weights)
        #
        #     # 🔹 Merge all inputs with portfolio weights
        #     for i in range(len(stock_tickers)):
        #         portfolio_records.append([
        #             selected_test_date,  # Store time period
        #             stock_tickers[i],
        #             sectors[i],
        #             predicted_probs[i],
        #             rolling_volatility[i],
        #             0.353392545,  # Confidence threshold lower bound
        #             0.662810753,  # Confidence threshold upper bound
        #             0.262342442,  # Sector constraint limit
        #             False,  # Annualized volatility flag
        #             portfolio_weights.loc[portfolio_weights["Stock"] == stock_tickers[i], "Final Weight"].values[0],
        #             bmk_weights_abs_wt.loc[bmk_weights_abs_wt["Stock"] == stock_tickers[i], "Final Weight"].values[0],
        #             bmk_weights_raw_pred.loc[bmk_weights_raw_pred["Stock"] == stock_tickers[i], "Final Weight"].values[0]
        #
        #             # Extract weight
        #         ])



        start += prediction_window
    #
    # portfolio_results_df = pd.DataFrame(portfolio_records, columns=[
    #     "Time Period", "Stock", "Sector", "Predicted Probability", "Rolling Volatility",
    #     "Confidence Threshold (Low)", "Confidence Threshold (High)", "Short Ratio",
    #     "Annualized Volatility", "Final Portfolio Weight", "BM Weight Absolute Weight Scale", "BM Weight Raw Preditor Scale"
    # ])
    #
    # # Save results to a CSV file
    # portfolio_results_df.to_csv("portfolio_results.csv", index=False)
    ###################################################################
    # UNCOMMENT FOR TRADING
    # Convert to DataFrame
    # Convert to DataFrame
    if train_model2==True:
        df_results = pd.DataFrame(results, columns=["Stock", "Phase", "Date", "Predicted Return", "Actual Return"])
        df_classification = pd.DataFrame(classification_results,
                                         columns=["Stock", "Phase", "Start Date", "End Date", "Predicted Binary",
                                                  "Predict Probability", "Actual Binary", "Stock Volatility",
                                                  "Next Period Return"])

        # Save results to CSV
        df_results.to_csv("stock_predictions_results.csv", index=False)
        df_classification.to_csv("stock_predictions_classification.csv", index=False)

        print("Results saved successfully!")
    elif train_model2==False:
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
        shap_metrics_results_df.to_csv("shap_value_metrics_export.csv", index=False)
        print("shap_value_metrics_export saved successfully!")

elif state == 'train_RL':

    # Load sample stock data (Replace with actual data loading)
    stock_data = pd.read_csv("sample_stock_data.csv")  # Assume CSV file with test data

    # Train the RL agent
    trained_model = train_rl_agent(stock_data, max_periods=213, num_episodes=5000)
    # trained_model = train_rl_agent(stock_data, max_periods=213, num_episodes=5000)

    model_save_path = "trained_portfolio_rl_agent.zip"
    trained_model.save(model_save_path)
    print(f"Trained model saved to {model_save_path}")

    # Load the trained model (Optional)
    loaded_model = PPO.load(model_save_path)
    print("Loaded trained model successfully!")

    # Get averaged optimal parameters
    M = 100  # Number of samples to generate
    avg_short_ratio, avg_confidence_threshold = get_average_optimal_parameters(trained_model, stock_data,
                                                                               num_periods=10, M=M)

    print("Averaged Optimized Short Ratio:", avg_short_ratio)
    print("Averaged Optimized Confidence Threshold Low:", avg_confidence_threshold[0])
    print("Averaged Optimized Confidence Threshold High:", avg_confidence_threshold[1])

    # Save averaged optimal parameters to CSV
    optimal_params_df = pd.DataFrame({
        'Short Ratio': [avg_short_ratio],
        'Confidence Threshold Low': [avg_confidence_threshold[0]],
        'Confidence Threshold High': [avg_confidence_threshold[1]]
    })
    optimal_params_df.to_csv("optimal_parameters.csv", index=False)

    print("Optimal parameters saved to optimal_parameters.csv")

elif state == 'SHAP_optimal_weight':
    file_path = "transformed_shap_timeseries_updated_sort.csv"
    df_shap = pd.read_csv(file_path, index_col=[0, 1])

    # Load the daily returns dataset
    daily_returns_path = "daily_returns_sort.csv"
    df_returns = pd.read_csv(daily_returns_path, parse_dates=["Date"], index_col="Date")

    # Extract ETFs
    global etfs
    etfs = list(set([col.split("_")[-1] for col in df_shap.columns if "mean_return" in col]))
    # global etfs
    # num_etfs = len(etfs)

    # Train agent with rolling 78-sample window
    window_size = 26*5
    num_samples = len(df_shap)
    out_of_sample_weights = []
    rebal_window = 26
    use_cash = True

    # for start in range(0, num_samples - window_size, window_size):
    for start in range(0, num_samples - window_size, rebal_window):
        end = start + window_size
        train_data = df_shap.iloc[start:end]
        env = make_vec_env(lambda: PortfolioTradingEnv(train_data, df_returns, initial_balance=1000, etfs=etfs, use_cash=True), n_envs=1)
        policy_kwargs = dict(
            net_arch=[256, 256, 128],  # Deep neural network
            activation_fn=torch.nn.ReLU  # Use ReLU activation
        )

        # Hyperparameter optimization
        ppo_params = {
            "policy": "MlpPolicy",
            "env": env,
            "verbose": 1,
            "learning_rate": 1e-4,  # Reduced learning rate for stability
            "n_steps": 2048,  # Higher steps for better learning
            "batch_size": 64,  # Adjust batch size
            "gamma": 0.995,  # Higher discount factor for long-term rewards
            "gae_lambda": 0.95,  # Adjust lambda for advantage estimation
            "clip_range": 0.2,  # Clipping range
            "ent_coef": 0.01,  # Encourage exploration,
            "policy_kwargs": policy_kwargs
        }

        model = PPO(**ppo_params)
        # Training with evaluation callback
        eval_callback = EvalCallback(env, best_model_save_path="./ppo_best_model", log_path="./logs", eval_freq=5000)
        model.learn(total_timesteps=100000, callback=eval_callback)

        # model = PPO("MlpPolicy", env, verbose=1)
        # model.learn(total_timesteps=10000)
        # model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=1e-4)
        # eval_callback = EvalCallback(env, best_model_save_path="./ppo_best_model", eval_freq=5000)
        # model.learn(total_timesteps=200000, callback=eval_callback)


        # Save trained model
        model_path = f"ppo_portfolio_model_{start}_{end}.zip"
        model.save(model_path)
        print(f"Model saved: {model_path}")

        # Generate out-of-sample weights for next rebal_window periods
        # Initialize out-of-sample weights as a list

        # Generate out-of-sample weights for next rebal_window periods
        test_data = df_shap.iloc[end:end + rebal_window]
        obs = test_data.values
        weights_list = []

        for _ in range(100):  # Generate 100 different weight samples
            batch_weights = []
            for row in obs:
                action, _ = model.predict(row, deterministic=False)

                # Ensure action is safe and normalized
                action = np.clip(action, 0, 1)
                if np.sum(action) == 0:
                    action = np.ones_like(action) / len(action)
                else:
                    action /= np.sum(action)

                batch_weights.append(action)

            weights_list.append(batch_weights)

        # Compute average weights across 100 runs
        avg_weights = np.mean(weights_list, axis=0)

        # Normalize each row again just in case
        avg_weights = np.array([w / np.sum(w) for w in avg_weights])

        # Collect start and end dates
        start_dates = df_shap.index.get_level_values(0)[end:end + rebal_window]
        end_dates = df_shap.index.get_level_values(1)[end:end + rebal_window]

        # Ensure lists are used before appending
        for i in range(len(avg_weights)):
            out_of_sample_weights.append([start_dates[i], end_dates[i]] + avg_weights[i].tolist())
        # Save results with appropriate column names
    out_of_sample_weights = np.squeeze(np.array(out_of_sample_weights))

    # Set column names depending on use_cash
    columns = ['Start Date', 'End Date'] + etfs + (["Cash"] if use_cash else [])

    # Create and save DataFrame
    out_of_sample_df = pd.DataFrame(out_of_sample_weights, columns=columns)
    out_of_sample_df.to_csv("out_of_sample_weights.csv", index=False)
    print("Training complete. Models and out-of-sample weights saved.")

