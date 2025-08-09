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


# Example usage
input_dir = "E:\XAI\RL_with_SHAP\pythonProject1\SVERL_icml_2023\portfolio_DRL"  # Use the specified directory
ticker_file = os.path.join(input_dir, "tickers.txt")  # Text file containing tickers, one per line
factor_file = os.path.join(input_dir, "FF_factors.csv")  # CSV file containing factor returns
start_date = "2020-01-01"
end_date = "2023-12-31"
output_csv = os.path.join(input_dir, "stock_prices.csv")
#
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

Nday_diff = 30  # Define the future prediction horizon for shift days
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

N = 60
sequence_length = 60
batch_size = 32
num_epochs = 100
learning_rate = 0.001
rolling_window_train = 180  # Reduce for faster adaptation
rolling_window_test = 60    # Reduce test window
prediction_window = 60      # Align prediction period with test window

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Record results
results = []
prediction_records = []
shapley_records = []
shapley_metrics_df = []
all_test_losses = []
test_results = []
prediction_results = []
combined_test_result = []

for stock in returns_raw.columns:
    print(f"Training for stock: {stock}")
    stock_returns = returns_raw_Y[[stock]]  # Select the stock column
    factors_raw_X = pd.concat([factors_raw, stock_returns], axis=1)
    # Rolling split: Train (M days), Test (N days), Predict (W days)
    total_samples = len(factors_raw_X)
    # Initialize the rolling split index
    start = 0

    while start + rolling_window_train + prediction_window + prediction_window <= total_samples:
        train_start = start
        train_end = start + rolling_window_train

        test_start = train_end
        test_end = test_start + rolling_window_test

        predict_start = test_end  # Predictions start right after training
        predict_end = predict_start + prediction_window

        # Extract the actual date indices
        train_dates = (factors_raw_X.iloc[train_start].name, factors_raw_X.iloc[train_end - 1].name)
        test_dates = (factors_raw_X.iloc[test_start].name, factors_raw_X.iloc[test_end - 1].name)
        predict_dates = (factors_raw_X.iloc[predict_start].name, factors_raw_X.iloc[predict_end - 1].name)

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

        # Avoid division by zero
        train_factors_std.replace(0, 1e-8, inplace=True)
        train_returns_std.replace(0, 1e-8, inplace=True)

        train_factors_norm = (train_factors - train_factors_mean) / train_factors_std
        train_returns_norm = (train_returns - train_returns_mean) / train_returns_std
        test_factors_norm = (test_factors - train_factors_mean) / train_factors_std
        test_returns_norm = (test_returns - train_returns_mean) / train_returns_std
        predict_factors_norm = (predict_factors - train_factors_mean) / train_factors_std
        predict_returns_norm = (predict_returns - train_returns_mean) / train_returns_std
        print(f'train_factors_norm dim: {train_factors_norm.shape}')
        print(f'train_returns_norm dim: {train_returns_norm.shape}')
        dataset = StockDataset(train_factors_norm, train_returns_norm, sequence_length, N=N)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # # Model initialization
        d_model = train_factors.shape[1]
        print(f'd_model: {d_model}')

        num_heads = min(4, d_model)  # Ensure num_heads does not exceed d_model and d_model is divisible by num_heads
        while d_model % num_heads != 0:
            num_heads -= 1
        # print(f"d_model: {d_model} num_heads: {num_heads} sequence_length: {sequence_length} and train factor return dim: {train_factors.shape[1]} {train_returns.shape[1]}")
        model = StockPredictAgent(d_model=d_model, num_heads=num_heads, sequence_length=sequence_length, N=N).to(device)
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

        # Convert to PyTorch tensors
        test_factors_tensor = torch.tensor(test_factors_norm.values, dtype=torch.float32).to(device).unsqueeze(0)
        predict_factors_tensor = torch.tensor(predict_factors_norm.values, dtype=torch.float32).to(device).unsqueeze(0)
        print(f'test_factors_tensor: {test_factors_tensor.shape}')
        # print(f'test_factors_tensor: {test_factors_tensor.shape}')
        # Run inference on test data
        model.eval()
        with torch.no_grad():
            test_predictions_norm = model(test_factors_tensor).cpu().numpy().flatten()
            # future_predictions_norm = model(predict_factors_tensor).cpu().numpy().flatten()

            # test_output = model(torch.tensor(test_factors_np, dtype=torch.float32).to(device))
            # print(f"Model output shape: {test_output.shape}")  # Expected: (1, 20) for 20 future returns
        # print(f'test_factors_tensor: {test_factors_tensor.shape}')
        # Revert normalization for test and predicted returns
        test_predictions = test_predictions_norm * train_returns_std.item() + train_returns_mean.item()

        #
        # # start SHAP
        # Prepare your input: (1 sample, 20 days, 6 features)
        # ######################## WRAP THIS AS FUNCTION
        # test_factors_np = test_factors_norm.values.astype('float32')
        # test_factors_np = test_factors_np.reshape(1, rolling_window_test,
        #                                           d_model)  # Reshaping for 1 sample with 20 days and 7 features
        # # print(f'test_factors_np dim: {test_factors_np.shape}')
        # #
        # # # Use a subset of the training data as the background dataset
        # background_data = torch.tensor(train_factors_norm.values.astype('float32')).to(device)
        # background_data = background_data.reshape(-1, rolling_window_test, d_model)  # Ensure same shape as model input
        # # print(f'background_data dim: {background_data.shape}')
        # # Ensure background_data is a tensor and avoid redundant torch.tensor() calls
        # if not isinstance(background_data, torch.Tensor):
        #     background_data = torch.tensor(background_data, dtype=torch.float32).to(device)
        # else:
        #     background_data = background_data.to(device)  # Just move to device without re-wrapping
        #
        # # Initialize SHAP GradientExplainer
        # explainer = shap.GradientExplainer(model, background_data)
        #
        # # Compute SHAP values for the test input
        # shap_values = explainer.shap_values(torch.tensor(test_factors_np, dtype=torch.float32).to(device),
        #                                     ranked_outputs=1)
        # # Aggregate SHAP values across outputs and time
        # # Verify SHAP values shape
        # # print(f"SHAP values shape: {shap_values[0].shape}")  # Expected: (1, 20, 6)
        # # print(f"test factor shape: {test_factors_norm.shape}")  # Expected: (1, 20, 6)
        #
        # ############# train using combined SHAP measures and test factor against binary test return
        # # Normalize SHAP values using the same mean and std as training factors
        # shap_values_squeezed = np.squeeze(shap_values[0], axis=(0, -1))  # Shape: (60, 6)
        # shap_values_normalized = (shap_values_squeezed - train_factors_mean.values) / train_factors_std.values
        #
        # # Concatenate normalized SHAP values with normalized test factors
        # test_factors_np = test_factors_norm.values.astype('float32')  # Shape: (60, 6)
        # augmented_features = np.concatenate([test_factors_np, shap_values_normalized], axis=1)  # Shape: (60, 12)
        # ########## END OF WRAPPING FUNCTION
        # Generate augmented features

        augmented_test_features = generate_augmented_features(
            test_factors_norm, train_factors_norm, train_factors_mean, train_factors_std,
            model, rolling_window_test, device
        )
        # Convert to PyTorch tensor if needed
        augmented_features_tensor = torch.tensor(augmented_test_features, dtype=torch.float32).to(device)

        print(f"Final Augmented Tensor Shape: {augmented_features_tensor.shape}")  # Expected: (rolling_window_test, d_model * 2)

        # Convert test_returns_norm into binary classification
        cumulative_return = np.prod(test_returns_norm + 1) - 1  # Aggregate 60-day return
        test_returns_binary = (cumulative_return > train_returns_mean).astype(int)  # 1 if positive, 0 if negative

        print(f"Cumulative Return over 60 days: {cumulative_return}")
        print(f"Binary Target (Up=1, Down=0): {test_returns_binary}")

        # Train the Model Using Augmented Features
        # sequence_length = 20
        # batch_size = 32
        # num_epochs = 25
        # learning_rate = 0.001
        d_model2 = augmented_test_features.shape[1]  # Number of features (6 original + 6 SHAP)
        num_heads = min(4, d_model2)
        while d_model2 % num_heads != 0:
            num_heads -= 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Length of augmented_test_features: {augmented_test_features.shape}")
        print(f"Length of test_returns_norm: {test_returns_norm.shape}")
        print(f"Length of sequence: {sequence_length}")

        # sequence_length2 = min(10, len(augmented_test_features))  # Ensure it's smaller than available data
        sequence_length2 = prediction_window
        # Create dataset and dataloader for training
        dataset2 = StockDatasetBinary(augmented_test_features, test_returns_norm, sequence_length2)
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
        # Convert augmented test features to tensor

        augmented_predict_features = generate_augmented_features(
            predict_factors_norm, train_factors_norm, train_factors_mean, train_factors_std,
            model, prediction_window, device
        )
        augmented_features_tensor_predict = torch.tensor(augmented_predict_features, dtype=torch.float32).to(device)
        # augmented_features_tensor_predict = torch.tensor(augmented_predict_features, dtype=torch.float32).to(device).unsqueeze(0)
        augmented_features_tensor_predict = augmented_features_tensor_predict[-10:]  # Select the last 10 time steps
        augmented_features_tensor_predict = augmented_features_tensor_predict.unsqueeze(0)  # Add batch dimension

        # Ensure input has (batch_size=1, sequence_length=60, d_model=12)
        # augmented_features_tensor_predict = augmented_features_tensor_predict.view(1, 60, 12)  # Reshape explicitly

        print(f"Fixed Shape: {augmented_features_tensor_predict.shape}")  # Should be


        # Get model predictions
        model_binary.eval()
        with torch.no_grad():
            return_predictions_prob = model_binary(augmented_features_tensor_predict).cpu().numpy().flatten()
            # return_test_prob = model_binary(augmented_features_tensor).cpu().numpy().flatten()


        # Convert probabilities to binary labels (0 = down, 1 = up)
        # Convert probability outputs to binary labels (0 = down, 1 = up)
        print(f'return_predictions_prob: {return_predictions_prob}')
        return_predictions_binary = (return_predictions_prob > 0.5).astype(int)
        #
        # # Display predictions
        # print(f"Predicted Binary Stock Returns: {return_predictions_binary}")

        # Print predictions
        # print(f"Predicted Stock Directions: {return_predictions_binary}")

        # ###### print SHAP RELATED METRICS
        # # shap_values_squeeze = shap_values[0].reshape(1,rolling_window_test,d_model)
        #
        # # 1. Mean Absolute SHAP Values (Feature Importance)
        # mean_abs_shap_values = np.mean(np.abs(shap_values[0]), axis=(0, 1))  # Shape: (6,)
        # # print(f'mean_abs_shap_values: {mean_abs_shap_values}')
        # # 2. SHAP Value Variance (Variability of Feature Contributions)
        # shap_value_variance = np.var(shap_values[0], axis=(0, 1))  # Shape: (6,)
        # # print(f'shap_value_variance: {shap_value_variance}')
        # # 3. SHAP Interaction Values (Feature Interdependencies)
        # # interaction_values = shap.TreeExplainer(model).shap_interaction_values(test_factors_tensor)
        #
        # # 4. Create DataFrame to Display Metrics
        # feature_importance_df = pd.DataFrame({
        #     'Date': test_factors.index[-1],
        #     'Feature': factors_raw_X.columns,
        #     'Mean_Abs_SHAP_Value': mean_abs_shap_values.flatten(),
        #     'SHAP_Value_Variance': shap_value_variance.flatten()
        #     # 'SHAP_Value_MEAN_DIV_VAR': mean_abs_shap_values.flatten() / shap_value_variance.flatten()
        #
        # })
        #
        # feature_importance_df_pivot = feature_importance_df.pivot(
        #         index='Date',
        #         columns='Feature',
        #         values=['Mean_Abs_SHAP_Value', 'SHAP_Value_Variance']
        #     ).reset_index()
        # # Sort by Feature Importance
        # # feature_importance_df = feature_importance_df.sort_values(by='Mean_Abs_SHAP_Value', ascending=False)
        #
        # # Flatten MultiIndex columns for readability
        # feature_importance_df_pivot.columns = ['_'.join(col).strip() if col[1] else col[0] for col in feature_importance_df_pivot.columns.values]
        #
        #
        #
        # print("Feature Importance Metrics (Mean SHAP & Variance):")
        # print(feature_importance_df_pivot.to_string(index=False))
        # # print(f'feature_importance_df: {feature_importance_df}')
        #
        # # # end shap
        # ###### print SHAP RELATED METRICS

        future_predictions = future_predictions_norm * train_returns_std.item() + train_returns_mean.item()
        actual_test_returns = test_returns.values.flatten()
        actual_predict_returns = predict_returns.values.flatten()
        # print(f'test_predictions dim:{test_predictions.shape} actual_test_returns dim: {actual_test_returns.shape}')

        # Compute loss for test phase
        test_loss = np.mean((test_predictions - actual_test_returns) ** 2)  # MSE Loss
        all_test_losses.append(test_loss)
        print(f'test_loss: {test_loss}')
        # Store test results
        combined_test_result = {
            "Stock": stock,
            "Test Dates": (test_start, test_end),
            "Predicted Returns (Test Set)": test_predictions.tolist(),
            "Actual Returns (Test Set)": actual_test_returns.tolist(),
            "Loss": test_loss
        }

        # Add SHAP feature importance metrics to the test results
        # combined_test_result.update(feature_importance_df_pivot.iloc[0].to_dict())

        # Append the combined results to the test_results list
        test_results.append(combined_test_result)

        # Store prediction results
        # prediction_results.append({
        #     "Stock": stock,
        #     "Prediction Dates": (predict_start, predict_end),
        #     "Predicted Returns": future_predictions.tolist(),
        #     "Actual Returns": actual_predict_returns.tolist()
        # })
        # Match dates with predictions and actual returns
        # date_range = pd.date_range(start=test_start, periods=len(actual_predict_returns), freq="D")

        for date, predicted, actual in zip(predict_returns.index, future_predictions, actual_predict_returns):
            prediction_results.append({
                "Stock": stock,
                "Date": date,
                "Predicted Return": predicted,
                "Actual Return": actual
            })
        print(f"Predictions for {stock} from {predict_start} to {predict_end} complete.")
        start += prediction_window

    # Print average test loss for this stock
    avg_test_loss = np.mean(all_test_losses)
    print(f"Average Test Loss for {stock}: {avg_test_loss:.6f}")
future_predictions_df = pd.DataFrame(prediction_results)

future_predictions_df.to_csv(f"{input_dir}/predicted_vs_actual.csv", index=False)
test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv(f"{input_dir}/test_result_output.csv", index=False)