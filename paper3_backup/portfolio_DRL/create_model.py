import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from SVERL_icml_2023.shapley import Shapley

import torch


class StockPredictWrapper:
    def __init__(self, model, device):
        """
        Wrapper for StockPredictAgent to work with SHAP and ensure input shape consistency.

        Parameters:
        - model: The trained PyTorch model (StockPredictAgent)
        - device: The computation device (CPU or CUDA)
        """
        self.model = model.to(device)
        self.device = device

    def __call__(self, factors):
        """
        Perform inference using the model, ensuring correct tensor conversion.

        Parameters:
        - factors: NumPy array of factor data

        Returns:
        - Model predictions as a NumPy array
        """

        # Convert input from NumPy to PyTorch tensor
        factors = torch.tensor(factors, dtype=torch.float32).to(self.device)

        # Ensure correct shape (batch_size, sequence_length, num_features)
        if len(factors.shape) == 2:  # If missing sequence dimension, assume seq_length=1
            factors = factors.unsqueeze(1)

        with torch.no_grad():
            predictions = self.model(factors).cpu().numpy()  # Ensure output is NumPy
        return predictions


import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(attn_output.size(0), -1, self.d_model)

        return self.W_o(attn_output)


class StockPredictAgent(nn.Module):
    def __init__(self, d_model, num_heads, sequence_length, N, dropout=0.1):
        super(StockPredictAgent, self).__init__()
        # self.pos_encoding = PositionalEncoding(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.fc = nn.Linear(sequence_length * d_model, N)
        self.dropout = nn.Dropout(dropout)

    def forward(self, factors):
        attn_output = self.attention(factors)
        batch_size, seq_length, feature_dim = attn_output.shape

        # print(f"Attention Output Shape: {attn_output.shape}")  # Debugging step

        flattened = attn_output.reshape(batch_size, seq_length * feature_dim)

        # print(f"Flattened Shape Before FC Layer: {flattened.shape}")  # Debugging step

        return self.fc(self.dropout(flattened))


class StockDataset(torch.utils.data.Dataset):
    def __init__(self, factors, returns, sequence_length, N):
        self.factors = factors.values  # Shape: (num_samples, num_features)
        self.returns = returns.values  # Shape: (num_samples, 1)
        self.sequence_length = sequence_length
        self.N = N  # Number of future days

    def __len__(self):
        return len(self.factors) - self.sequence_length - self.N + 1

    def __getitem__(self, idx):
        factors_seq = self.factors[idx:idx + self.sequence_length]  # (sequence_length, num_features)
        target = self.returns[idx + self.sequence_length: idx + self.sequence_length + self.N]  # Future returns

        # 🚀 **Force Correct Shape: Ensure (sequence_length, num_features)**
        if len(factors_seq.shape) == 3:
            factors_seq = factors_seq.squeeze(0)  # Remove batch dim if present

        # 🚀 **Force X to be exactly (sequence_length, num_features)**
        factors_seq = torch.tensor(factors_seq, dtype=torch.float32).reshape(self.sequence_length, -1)
        target = torch.tensor(target, dtype=torch.float32)

        return factors_seq, target


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class StockDatasetBinary(torch.utils.data.Dataset):
    def __init__(self, features, returns_norm, sequence_length):
        """
        Parameters:
        - features (np.array): Augmented feature set (X variables).
        - returns_norm (pd.DataFrame): Normalized returns (Y variable).
        - sequence_length (int): Length of the input sequence.
        """
        self.features = features
        self.returns_norm = returns_norm.values

        # Convert returns into cumulative binary classification
        self.binary_labels = ((np.prod(self.returns_norm + 1, axis=1) - 1) > 0).astype(int)

        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x_seq = self.features[idx:idx + self.sequence_length]  # X variables
        y_label = self.binary_labels[idx + self.sequence_length - 1]  # Binary Y variable (0 or 1)
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.float32)


class StockPredictAgentBinary(nn.Module):
    def __init__(self, d_model, num_heads, sequence_length, dropout=0.1):
        super(StockPredictAgentBinary, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Fully connected layer with 1 output (sigmoid activation for binary classification)
        self.fc = nn.Linear(sequence_length * d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, factors):
        attn_output = self.attention(factors)
        batch_size, seq_length, feature_dim = attn_output.shape

        # Flatten the output for the classification layer
        flattened = attn_output.reshape(batch_size, seq_length * feature_dim)

        # Output a probability using sigmoid activation
        return torch.sigmoid(self.fc(self.dropout(flattened)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Ensure d_model is even
        if d_model % 2 != 0:
            d_model += 1

        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Input dimension {x.shape[-1]} does not match expected d_model {self.d_model}")
        return x + self.pe[:, :x.size(1), :].to(x.device)


import torch.optim as optim


def train_model_with_logging(model, dataloader, optimizer, criterion, scheduler, num_epochs, device):
    """
    Train the model while logging loss values at each epoch.

    Parameters:
    model (nn.Module): The model to train.
    dataloader (DataLoader): DataLoader for training data.
    num_epochs (int): Number of training epochs.
    device (torch.device): Device to train the model on (CPU or GPU).

    Returns:
    dict: Training loss per epoch.
    """
    model.to(device)
    model.train()

    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx, (factors, target) in enumerate(dataloader):
            factors, target = factors.to(device), target.to(device)

            optimizer.zero_grad()  # Reset gradients
            predictions = model(factors)  # Forward pass

            target = target.view_as(predictions)
            loss = criterion(predictions, target)  # Compute loss

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss encountered at epoch {epoch + 1}, batch {batch_idx + 1}")
                return epoch_losses

            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            total_loss += loss.item()

        # Adjust learning rate
        scheduler.step()

        # Compute average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)

        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return epoch_losses


def train_model(model, dataloader, optimizer, criterion, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for factors, target in dataloader:
            factors, target = factors.to(device), target.to(device).unsqueeze(1)  # Ensure correct shape

            optimizer.zero_grad()
            output = model(factors)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


# def train_model(model, dataloader, optimizer, criterion, num_epochs, device):
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0.0
#         for factors, returns, target in dataloader:
#             factors, returns, target = factors.to(device), returns.to(device), target.to(device)
#
#             optimizer.zero_grad()
#             predictions = model(factors, returns)
#             # loss = criterion(predictions, target)
#             loss = criterion(predictions, target.squeeze(-1))
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")


# Function to calculate and record Shapley metrics
def calculate_shapley_metrics(shapley_values, test_factors):
    """
    Calculate and return useful Shapley metrics.

    Parameters:
    shapley_values (np.array): Shapley values calculated for each factor.
    test_factors (pd.DataFrame): DataFrame containing the test factors.

    Returns:
    dict: Dictionary containing metrics including mean absolute Shapley values, variance, and top factors per sample.
    """
    shapley_df = pd.DataFrame(shapley_values, columns=test_factors.columns, index=test_factors.index)

    # Mean Absolute Shapley Values
    mean_abs_shap = shapley_df.abs().mean(axis=0)

    # Variance of Shapley Values
    shapley_variance = shapley_df.var(axis=0)

    # Top Factors Per Sample
    top_factors_per_sample = shapley_df.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)

    return {
        "shapley_df": shapley_df,
        "mean_abs_shap": mean_abs_shap,
        "shapley_variance": shapley_variance,
        "top_factors_per_sample": top_factors_per_sample
    }


# Define a wrapper for the StockPredictAgent to work with SHAP

import shap
import torch
import numpy as np


def generate_augmented_features_testphase(test_factors_norm, train_factors_norm, train_factors_mean, train_factors_std,
                                          model, rolling_window_test, device, return_metrics = False):
    """
    Generate augmented feature set by combining test features with SHAP values.

    Parameters:
    - test_factors_norm (pd.DataFrame): Normalized test factors (X variables)
    - train_factors_norm (pd.DataFrame): Normalized training factors (used for SHAP background data)
    - train_factors_mean (pd.Series): Mean values of training factors (for normalization)
    - train_factors_std (pd.Series): Standard deviation values of training factors (for normalization)
    - model (torch.nn.Module): Trained model for SHAP value computation
    - rolling_window_test (int): Sequence length for test samples
    - device (torch.device): Device to run SHAP computations on (CPU/GPU)

    Returns:
    - augmented_features (np.array): Combined test features and SHAP values (shape: [num_samples, num_features * 2])
    """

    # Step 1: Convert test factors to NumPy and reshape
    test_factors_np = test_factors_norm.values.astype('float32').reshape(1, rolling_window_test, -1)
    d_model = test_factors_np.shape[2]  # Number of features

    # Step 2: Prepare background data for SHAP (using training factors)
    background_data = train_factors_norm.values.astype('float32').reshape(-1, rolling_window_test, d_model)
    background_data_tensor = torch.tensor(background_data, dtype=torch.float32).to(device)

    # Step 3: Ensure background_data is a PyTorch tensor
    if not isinstance(background_data, torch.Tensor):
        background_data_tensor = torch.tensor(background_data, dtype=torch.float32).to(device)

    # Step 4: Initialize SHAP GradientExplainer
    explainer = shap.GradientExplainer(model, background_data_tensor)

    # Step 5: Compute SHAP values for test features
    shap_values = explainer.shap_values(torch.tensor(test_factors_np, dtype=torch.float32).to(device), ranked_outputs=1)

    # Step 6: Process SHAP values (squeeze unnecessary dimensions)
    shap_values_squeezed = np.squeeze(shap_values[0], axis=(0, -1))  # Shape: (rolling_window_test, d_model)

    # Step 7: Normalize SHAP values using training statistics
    shap_values_normalized = (shap_values_squeezed - train_factors_mean.values) / train_factors_std.values

    # Step 8: Concatenate normalized SHAP values with test features
    augmented_features = np.concatenate([test_factors_np[0], shap_values_normalized],
                                        axis=1)  # Shape: (rolling_window_test, d_model * 2)

    if not return_metrics:
        return augmented_features

    # Compute SHAP metrics
    shap_metrics = {
        "mean_abs_shap": np.abs(shap_values_squeezed).mean(axis=0),
        "mean_shap": shap_values_squeezed.mean(axis=0),
        "shap_std": np.var(shap_values_squeezed, axis=0),
        "mean_over_std": shap_values_squeezed.mean(axis=0) / np.std(shap_values_squeezed, axis=0),
        "mean_abs_over_std": np.abs(shap_values_squeezed).mean(axis=0) / np.std(shap_values_squeezed, axis=0),
        "feature_importance_ranking": np.argsort(-np.abs(shap_values_squeezed).mean(axis=0)),
        "shap_value_distribution": shap_values_squeezed,
        "top_factors_per_sample": np.argsort(-np.abs(shap_values_squeezed), axis=1)[:, :3],
        "shap_interaction_values": explainer.shap_interaction_values(background_data_tensor) if hasattr(explainer,  'shap_interaction_values') else None,
        "cumulative_shap_impact": np.sum(np.abs(shap_values_squeezed), axis=1),
        # "shap_feature_correlation": np.corrcoef(test_factors_np.T, shap_values_squeezed.T)[:d_model, d_model:],
        "shap_trend": np.mean(shap_values_squeezed, axis=1),
    }

    metrics_flat = {}
    for key in ["mean_abs_shap", "mean_shap", "shap_std", "mean_over_std","mean_abs_over_std", "feature_importance_ranking"]:
        for i, value in enumerate(shap_metrics[key]):
            metrics_flat[f"{key}_{i + 1}"] = value
    shap_metrics_df = pd.DataFrame([metrics_flat])

    return augmented_features, shap_metrics_df

def generate_augmented_features(train_factors_norm, train_factors_mean, train_factors_std, model,
                                rolling_window_test,
                                device, return_metrics=False):
    """
    Generate augmented feature set by combining training features with SHAP values and compute relevant SHAP metrics.

    Parameters:
    - train_factors_norm (pd.DataFrame): Normalized training factors (X variables).
    - train_factors_mean (pd.Series): Mean values of training factors (for normalization).
    - train_factors_std (pd.Series): Standard deviation values of training factors (for normalization).
    - model (torch.nn.Module): Trained model for SHAP value computation.
    - rolling_window_test (int): Sequence length for background data.
    - device (torch.device): Device to run SHAP computations on (CPU/GPU).
    - return_metrics (bool): If True, returns SHAP metrics in addition to augmented features.

    Returns:
    - augmented_features (np.array): Combined training features and SHAP values.
    - shap_metrics (dict, optional): Dictionary containing computed SHAP metrics.
    """

    # ✅ Step 1: Convert training factors to NumPy
    train_factors_np = train_factors_norm.values.astype('float32')
    num_samples, d_model = train_factors_np.shape  # Extract number of samples and feature count
    background_data = train_factors_norm.values.astype('float32').reshape(-1, rolling_window_test, d_model)
    background_data_tensor = torch.tensor(background_data, dtype=torch.float32).to(device)

    # ✅ Debugging SHAP input shape
    print(
        f"✅ Background Data Shape Before SHAP: {background_data_tensor.shape}")  # Expected: (1, rolling_window_test, 6)

    # ✅ Step 3: Initialize SHAP GradientExplainer using training data
    explainer = shap.GradientExplainer(model, background_data_tensor)

    # ✅ Step 4: Compute SHAP values for the training set
    shap_values = explainer.shap_values(torch.tensor(background_data, dtype=torch.float32).to(device), ranked_outputs=1)

    # ✅ Step 5: Debug SHAP value shape before squeezing
    print(f"🚀 SHAP Raw Output Shape: {shap_values[0].shape}")  # Example: (1, rolling_window_test, 6, 1)

    # ✅ Step 6: Remove unnecessary dimensions
    shap_values_squeezed = shap_values[0].squeeze(-1)  # Removes last dim (1), result: (1, rolling_window_test, 6)

    # ✅ Step 7: Resize SHAP values to match (180, 6)
    shap_values_resized = np.resize(shap_values_squeezed, (num_samples, d_model))  # Shape: (180, 6)
    print(f"✅ SHAP Resized Shape: {shap_values_resized.shape}")  # Should be (180, 6)

    # ✅ Step 8: Normalize SHAP values using training statistics no need to normalize shap_values_resized
    shap_values_normalized = shap_values_resized
    # shap_values_normalized = (shap_values_resized - train_factors_mean.values) / train_factors_std.values
    # print(f'shap_values_resized={shap_values_resized} and shap_values_normalized={shap_values_normalized}')
    # ✅ Step 9: Concatenate normalized SHAP values with training features
    augmented_features = np.concatenate([train_factors_np, shap_values_normalized], axis=1)

    # Concatenate normalized SHAP values with training features
    augmented_features = np.concatenate([train_factors_np, shap_values_normalized], axis=1)

    if not return_metrics:
        return augmented_features

    # Compute SHAP metrics
    shap_metrics = {
        "mean_abs_shap": np.abs(shap_values_resized).mean(axis=0),
        "mean_shap": shap_values_resized.mean(axis=0),
        "shap_std": np.var(shap_values_resized, axis=0),
        "mean_over_std": shap_values_resized.mean(axis=0) / np.std(shap_values_resized, axis=0),
        "mean_abs_over_std":np.abs(shap_values_resized).mean(axis=0) / np.std(shap_values_resized, axis=0),
        "feature_importance_ranking": np.argsort(-np.abs(shap_values_resized).mean(axis=0)),
        "shap_value_distribution": shap_values_resized,
        "top_factors_per_sample": np.argsort(-np.abs(shap_values_resized), axis=1)[:, :3],
        "shap_interaction_values": explainer.shap_interaction_values(background_data_tensor) if hasattr(explainer, 'shap_interaction_values') else None,
        "cumulative_shap_impact": np.sum(np.abs(shap_values_resized), axis=1),
        "shap_feature_correlation": np.corrcoef(train_factors_np.T, shap_values_resized.T)[:d_model, d_model:],
        "shap_trend": np.mean(shap_values_resized, axis=1),
    }

    metrics_flat = {}
    for key in ["mean_abs_shap", "mean_shap", "shap_std", "mean_over_std","mean_abs_over_std", "feature_importance_ranking"]:
        for i, value in enumerate(shap_metrics[key]):
            metrics_flat[f"{key}_{i + 1}"] = value
    shap_metrics_df = pd.DataFrame([metrics_flat])

    return augmented_features, shap_metrics_df


import numpy as np
import pandas as pd


def compute_portfolio_weights_old(stock_tickers, predicted_probs, rolling_volatility, sectors,
                              confidence_threshold=(0.4, 0.6), sector_limit=0.3, annualize_volatility=False):
    """
    Compute portfolio weights using probability-based allocation, inverse volatility adjustment,
    and sector constraints.

    Parameters:
    - stock_tickers (list): List of stock tickers.
    - predicted_probs (np.array): Predicted probability of stock moving up.
    - rolling_volatility (np.array): Rolling volatility (daily or annualized).
    - sectors (list): List of sector classifications for each stock.
    - confidence_threshold (tuple): Threshold (low, high) to filter out uncertain predictions.
    - sector_limit (float): Maximum allowed weight per sector (default = 30%).
    - annualize_volatility (bool): Whether to annualize daily volatility using sqrt(252).

    Returns:
    - pd.DataFrame: Portfolio allocation with normalized weights.
    """

    # Convert probability into raw weights (scale from [-1,1])
    raw_weights = 2 * (predicted_probs - 0.5)

    # Apply confidence threshold: Ignore stocks with low confidence
    confidence_mask = (predicted_probs > confidence_threshold[1]) | (predicted_probs < confidence_threshold[0])
    filtered_weights = raw_weights * confidence_mask  # Zero out low-confidence predictions

    # Annualize volatility if required
    if annualize_volatility:
        rolling_volatility = rolling_volatility * np.sqrt(252)

    # Apply Volatility Adjustment: Wi,Tadjusted = Wi,T / σi
    adjusted_weights = filtered_weights / rolling_volatility  # Scale by inverse volatility

    # Normalize weights to sum to 1
    abs_sum = np.sum(np.abs(adjusted_weights))
    normalized_weights = adjusted_weights / abs_sum if abs_sum > 0 else adjusted_weights

    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame({
        "Stock": stock_tickers,
        "Sector": sectors,
        "Probability": predicted_probs,
        "Raw Weight": raw_weights,
        "Volatility": rolling_volatility,
        "Adjusted Weight": adjusted_weights,
        "Normalized Weight": normalized_weights
    })

    # Apply Sector Constraints: Ensure no sector > sector_limit (default 30%)
    sector_weights = portfolio_df.groupby("Sector")["Normalized Weight"].sum()
    sector_excess = sector_weights[sector_weights > sector_limit] - sector_limit  # Find sectors exceeding limit

    if not sector_excess.empty:
        for sector, excess_weight in sector_excess.items():
            print(f"Reducing {sector} exposure by {excess_weight:.2%}")

            # Reduce the sector weights proportionally
            sector_stocks = portfolio_df[portfolio_df["Sector"] == sector]
            total_sector_weight = sector_stocks["Normalized Weight"].sum()

            # Scale down weights within the sector
            portfolio_df.loc[portfolio_df["Sector"] == sector, "Normalized Weight"] -= (
                    sector_stocks["Normalized Weight"] / total_sector_weight * excess_weight
            )

        # Re-normalize after sector adjustment
        final_abs_sum = np.sum(np.abs(portfolio_df["Normalized Weight"]))
        portfolio_df["Normalized Weight"] /= final_abs_sum

    return portfolio_df[["Stock", "Sector", "Probability", "Normalized Weight"]]


import pandas as pd
import numpy as np


def compute_portfolio_weights_BMK(stock_tickers, predicted_probs, rolling_volatility, sectors,
                              confidence_threshold=(0.4, 0.6), short_ratio=0.25,
                              annualize_volatility=False, raw_weight_version =1):
    """
    Compute optimal portfolio weights for long-short portfolio ensuring total weight sums to 100%
    while maintaining the short ratio.

    Parameters:
    - stock_tickers (list): List of stock tickers.
    - predicted_probs (np.array): Predicted probability of stock moving up.
    - rolling_volatility (np.array): Rolling volatility (daily or annualized).
    - sectors (list): List of sector classifications for each stock.
    - confidence_threshold (tuple): Threshold (low, high) to filter out uncertain predictions.
    - short_ratio (float): Proportion of short exposure (between 0 and 50%).
    - annualize_volatility (bool): Whether to annualize daily volatility using sqrt(252).

    Returns:
    - pd.DataFrame: Portfolio allocation with normalized long-short weights.
    """
    # Convert probability into raw weights for long-short portfolio
    if (raw_weight_version ==1):
        raw_weights = abs(2 * (predicted_probs - 0.5))
    elif (raw_weight_version ==2):
        raw_weights = predicted_probs

    # Apply confidence threshold: Only include confident stocks
    confidence_mask = (predicted_probs >= confidence_threshold[1]) | (predicted_probs <= confidence_threshold[0])
    # print(f'confidence_mask = {confidence_mask}')
    filtered_weights = raw_weights * confidence_mask  # Zero out low-confidence predictions
    # print(f'filtered_weights = {filtered_weights}')

    # Check if no stock qualifies
    if np.all(filtered_weights == 0):
        raise ValueError("No stocks passed the confidence threshold; cannot generate portfolio.")

    # Annualize volatility if required
    if annualize_volatility:
        rolling_volatility = rolling_volatility * np.sqrt(252)

    # Apply Volatility Adjustment: Wi,Tadjusted = Wi,T / σi
    adjusted_weights = filtered_weights / rolling_volatility  # Scale by inverse volatility

    # Separate long and short portfolios based on short ratio
    long_weights = np.where(adjusted_weights > 0, adjusted_weights, 0)
    short_weights = np.where(adjusted_weights < 0, adjusted_weights, 0)

    # Normalize long and short weights so they sum to the target allocations
    long_sum = np.sum(long_weights)
    short_sum = np.abs(np.sum(short_weights))

    print(f'long_weights={long_weights}, short_weights={short_weights}, {long_sum} and {short_sum}')
    if long_sum > 0:
        long_weights *= (1 + short_ratio) / long_sum  # Scale long portfolio to (1 + short_ratio)
    if short_sum > 0:
        short_weights *= short_ratio / short_sum  # Scale short portfolio to -short_ratio

    # Compute final portfolio weights (long + short should sum to 100%)
    final_weights = long_weights + short_weights
    # final_weights /= np.sum(np.abs(final_weights))  # Normalize to ensure exact sum of 100%

    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame({
        "Stock": stock_tickers,
        "Sector": sectors,
        "Probability": predicted_probs,
        "Raw Weight": raw_weights,
        "Volatility": rolling_volatility,
        "Adjusted Weight": adjusted_weights,
        "Final Weight": final_weights
    })

    # Validate constraints
    total_portfolio_weight = np.sum(portfolio_df["Final Weight"])
    actual_short_ratio = np.abs(
        np.sum(portfolio_df[portfolio_df["Final Weight"] < 0]["Final Weight"])) / total_portfolio_weight

    assert np.isclose(total_portfolio_weight, 1.0,
                      atol=1e-6), f"Error: Total Portfolio Weight is {total_portfolio_weight}, should be 1.0."
    assert np.isclose(actual_short_ratio, short_ratio,
                      atol=1e-6), f"Error: Actual Short Ratio is {actual_short_ratio}, expected {short_ratio}."

    return portfolio_df[["Stock", "Sector", "Probability", "Final Weight"]]


def compute_portfolio_weights_per_period(df, confidence_threshold=(0.4, 0.6), short_ratio=0.25, annualize_volatility=False):
    """
    Compute optimal portfolio weights for long-short portfolio ensuring total weight sums to 100%
    while maintaining the short ratio.

    Parameters:
    - stock_tickers (list): List of stock tickers.
    - predicted_probs (np.array): Predicted probability of stock moving up.
    - rolling_volatility (np.array): Rolling volatility (daily or annualized).
    - sectors (list): List of sector classifications for each stock.
    - confidence_threshold (tuple): Threshold (low, high) to filter out uncertain predictions.
    - short_ratio (float): Proportion of short exposure (between 0 and 50%).
    - annualize_volatility (bool): Whether to annualize daily volatility using sqrt(252).

    Returns:
    - pd.DataFrame: Portfolio allocation with normalized long-short weights.
    """
    portfolio_results = []


    for period, period_data in df.groupby("End Date"):
        stock_tickers = period_data["Stock"].values
        predicted_probs = period_data["Predict Probability"].values
        rolling_volatility = period_data["Volatility"].values
        sectors = period_data["Sector"].values

        # Convert probability into raw weights for long-short portfolio
        raw_weights = 2 * (predicted_probs - 0.5)

        # Apply confidence threshold: Only include confident stocks
        confidence_mask = (predicted_probs >= confidence_threshold[1]) | (predicted_probs <= confidence_threshold[0])
        filtered_weights = raw_weights * confidence_mask  # Zero out low-confidence predictions

        # Skip period if no stocks qualify
        if np.all(filtered_weights == 0):
            continue

        # Annualize volatility if required
        if annualize_volatility:
            rolling_volatility *= np.sqrt(252)

        # Apply Volatility Adjustment: Wi,Tadjusted = Wi,T / σi
        adjusted_weights = filtered_weights / rolling_volatility  # Scale by inverse volatility

        # Separate long and short portfolios based on short ratio
        long_weights = np.where(adjusted_weights > 0, adjusted_weights, 0)
        short_weights = np.where(adjusted_weights < 0, adjusted_weights, 0)

        # Normalize long and short weights so they sum to the target allocations
        long_sum = np.sum(long_weights)
        short_sum = np.abs(np.sum(short_weights))

        if long_sum > 0:
            long_weights *= (1 + short_ratio) / long_sum  # Scale long portfolio to (1 + short_ratio)
        if short_sum > 0:
            short_weights *= short_ratio / short_sum  # Scale short portfolio to -short_ratio

        # Compute final portfolio weights (long + short should sum to 100%)
        final_weights = long_weights + short_weights

        # Store results
        period_portfolio = pd.DataFrame({
            "End Date": period,
            "Stock": stock_tickers,
            "Sector": sectors,
            "Probability": predicted_probs,
            "Raw Weight": raw_weights,
            "Volatility": rolling_volatility,
            "Adjusted Weight": adjusted_weights,
            "Final Weight": final_weights
        })

        portfolio_results.append(period_portfolio)

    # Combine all periods into a single DataFrame
    portfolio_df = pd.concat(portfolio_results, ignore_index=True)
    return portfolio_df
    # # Validate constraints
    # total_portfolio_weight = np.sum(portfolio_df["Final Weight"])
    # assert np.isclose(total_portfolio_weight, 1.0,
    #                   atol=1e-6), f"Error: Total Portfolio Weight is {total_portfolio_weight}, should be 1.0."
    #
    # actual_short_ratio = np.abs(np.sum(portfolio_df[portfolio_df["Final Weight"] < 0]["Final Weight"])) / total_portfolio_weight
    # assert np.isclose(actual_short_ratio, short_ratio,
    #                   atol=1e-6), f"Error: Actual Short Ratio is {actual_short_ratio}, expected {short_ratio}."
    # # print(f'final weight : {portfolio_df[["Final Weight"]]}')
    # return portfolio_df[["Stock", "Sector", "Probability", "Final Weight"]]

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import shimmy
import gymnasium as gym
import stable_baselines3

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces

class PortfolioOptimizationEnv(gym.Env):
    def __init__(self, stock_data, num_periods):
        super(PortfolioOptimizationEnv, self).__init__()
        self.stock_data = stock_data.sort_values(by=['End Date']).copy()  # Sort by time
        self.num_periods = num_periods
        self.current_step = 0
        self.portfolio_value = 1000  # Initial portfolio value

        # Dynamically determine min and max probability values
        # self.min_prob = stock_data['Predict Probability'].min()
        # self.max_prob = stock_data['Predict Probability'].max()

        # Action space parameters: Short Ratio, Confidence Threshold Low, Confidence Threshold High
        self.low = np.array([0.0, 0.2, 0.5])  # Short Ratio from 0% to 50%
        self.high = np.array([0.5, 0.5, 0.8])

        # Define action space in the normalized range of [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space: Dropping non-numeric columns
        num_features = stock_data.drop(columns=['Stock', 'Sector', 'Phase', 'Start Date', 'End Date']).shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1000  # Reset portfolio value to initial
        obs = self.stock_data.drop(columns=['Stock', 'Sector', 'Phase', 'Start Date', 'End Date']).iloc[
            self.current_step]
        return obs

    def rescale_action(self, action):
        """
        Rescale action from [-1, 1] range back to the original range [low, high].
        """
        rescaled_action = 0.5 * (np.clip(action, -1, 1) + 1) * (self.high - self.low) + self.low
        return np.clip(rescaled_action, self.low, self.high)

    import random

    def step(self, action):
        max_attempts = 10  # Maximum retries if no stocks qualify
        attempt = 0
        success = False

        while attempt < max_attempts:
            try:
                # Rescale action from [-1, 1] to original range
                action = self.rescale_action(action)
                short_ratio = action[0]
                confidence_threshold = (action[1], action[2])
                # print(f'Attempt {attempt + 1}: confidence_threshold = {confidence_threshold}, short_ratio = {short_ratio}')

                # Compute portfolio weights
                portfolio_weights = compute_portfolio_weights(
                    stock_tickers=self.stock_data['Stock'].values,
                    predicted_probs=self.stock_data['Predict Probability'].values,
                    rolling_volatility=self.stock_data['Volatility'].values,
                    sectors=self.stock_data['Sector'].values,
                    confidence_threshold=confidence_threshold,
                    short_ratio=short_ratio
                )

                success = True
                break  # Exit loop if portfolio generation is successful

            except ValueError as e:
                print(f"Warning: {e} - Retrying with a new random action.")
                action = np.random.uniform(-1, 1, size=self.action_space.shape)  # Generate new random action
                attempt += 1
                print(f'-----------------add one more attempt----------------------')

        if not success:
            raise RuntimeError("Error: No valid portfolio found after 10 attempts. Training aborted.")

        # Compute portfolio return
        portfolio_return = (portfolio_weights['Final Weight'] * self.stock_data['Next Period Return']).sum()

        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)

        # Reward is the updated portfolio value
        reward = self.portfolio_value

        self.current_step += 1
        done = self.current_step >= self.num_periods - 1

        next_state = self.stock_data.drop(columns=['Stock', 'Sector', 'Phase', 'Start Date', 'End Date']).iloc[
            self.current_step] if not done else np.zeros(self.observation_space.shape)

        return next_state, reward, done, {}


# Train the RL agent incrementally
def train_rl_agent_old(stock_data, max_periods=100, num_episodes=5000, N=100):
    best_model = None
    stock_data = stock_data[-N:]
    for num_periods in range(1, min(max_periods, len(stock_data)) + 1):
        env = PortfolioOptimizationEnv(stock_data[:num_periods], num_periods)
        env = DummyVecEnv([lambda: env])

        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
        model.learn(total_timesteps=num_episodes)

        best_model = model  # Store best model at each step

    best_model.save("portfolio_rl_agent")
    return best_model

def train_rl_agent(stock_data, max_periods=100, num_episodes=5000):
    """
    Train the reinforcement learning agent by progressively increasing the data from the earliest available date
    in the `End Date` column to the full dataset.

    Parameters:
    - stock_data (pd.DataFrame): The dataset containing stock features.
    - max_periods (int): The maximum number of periods to train on.
    - num_episodes (int): The total timesteps to train each model.

    Returns:
    - The trained RL model.
    """
    best_model = None

    # Convert 'End Date' to datetime format for filtering
    stock_data['End Date'] = pd.to_datetime(stock_data['End Date'])

    # Sort the dataset by End Date (ascending order)
    stock_data = stock_data.sort_values(by="End Date")

    # Get unique sorted End Dates
    unique_dates = stock_data["End Date"].unique()

    for i, end_date in enumerate(unique_dates):
        # Filter stock data up to the current `end_date`
        filtered_stock_data = stock_data[stock_data["End Date"] <= end_date]

        # Ensure that the dataset has at least one period
        if len(filtered_stock_data) < 1:
            continue

        num_periods = len(filtered_stock_data)
        num_periods = min(max_periods, num_periods)  # Ensure it doesn't exceed max_periods

        print(f"Training with data up to: {end_date}, using {num_periods} periods.")

        # Create RL environment
        env = PortfolioOptimizationEnv(filtered_stock_data, num_periods)
        env = DummyVecEnv([lambda: env])

        # Initialize PPO model
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)

        # Train the model
        model.learn(total_timesteps=num_episodes)

        best_model = model  # Store the last trained model

    # Save the best trained model
    best_model.save("portfolio_rl_agent")

    return best_model




# Load trained agent and get optimal parameters
def get_average_optimal_parameters(model, stock_data, num_periods, M=10):
    """
    Generates M samples of optimal parameters from the trained model
    and computes the average for each parameter.

    Parameters:
    - model: Trained RL model
    - stock_data: DataFrame containing stock data
    - num_periods: Number of periods to use for evaluation
    - M: Number of samples to generate

    Returns:
    - Tuple of averaged short ratio and confidence thresholds
    """
    short_ratios = []
    confidence_thresholds = []

    for _ in range(M):
        env = PortfolioOptimizationEnv(stock_data[:num_periods], num_periods)
        env = DummyVecEnv([lambda: env])
        obs = env.reset()

        action, _ = model.predict(obs)
        action = np.array(action).flatten()  # Ensure it's a 1D array

        low = env.envs[0].low
        high = env.envs[0].high

        # Rescale action from [-1, 1] to original range
        rescaled_action = 0.5 * (np.clip(action, -1, 1) + 1) * (high - low) + low
        rescaled_action = np.clip(rescaled_action, low, high)

        # Debugging: Print action shape
        print(f"Raw Action: {action}, Rescaled Action: {rescaled_action}, Shape: {rescaled_action.shape}")

        # Ensure rescaled_action has at least 3 elements
        if rescaled_action.shape[0] < 3:
            raise ValueError(f"Unexpected action shape: {rescaled_action.shape}. Expected at least 3 values.")

        # Append rescaled values
        short_ratios.append(rescaled_action[0])  # Short Ratio
        confidence_thresholds.append([rescaled_action[1], rescaled_action[2]])  # Confidence thresholds

    # Compute the average across M samples
    avg_short_ratio = np.mean(short_ratios)
    avg_confidence_threshold = np.mean(confidence_thresholds, axis=0)

    return avg_short_ratio, tuple(avg_confidence_threshold)


class PortfolioTradingEnv(gym.Env):
    def __init__(self, data, returns, initial_balance=1000, etfs=None):
        super(PortfolioTradingEnv, self).__init__()
        self.data = data
        self.returns = returns
        self.initial_balance = initial_balance
        self.current_step = 0
        self.etfs = etfs if etfs else []
        self.num_etfs = len(self.etfs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1.0, shape=(12,), dtype=np.float32)
        self.portfolio_value = self.initial_balance
        self.weights = np.ones(self.num_etfs) / self.num_etfs

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.weights = np.ones(self.num_etfs) / self.num_etfs
        return self.data.iloc[self.current_step].values

    def step(self, action):
        # print(f"   === Step  {self.current_step}   Debuginfo == = ")
        action = np.nan_to_num(np.array(action, dtype=np.float32))
        # print(f"Raw Action: {action}")
        if np.sum(action) == 0:
            action = np.ones(self.num_etfs) / self.num_etfs  # Reset to equal weights if all values are zero

        action = np.clip(action, 0, 1)
        action /= np.sum(action)
        self.weights = action

        start_date, end_date = self.data.index[self.current_step]
        # print(f"Start Date: {start_date}, End Date: {end_date}")
        # print(f"Current Return: {self.returns.loc[start_date:end_date,:].values}")
        # print(f'self.etfs={self.etfs}')
        period_returns = self.returns.loc[start_date:end_date, self.etfs].values
        # print(f"Period Returns: {period_returns}")
        portfolio_values = [self.portfolio_value]
        for daily_return in period_returns:
            self.portfolio_value *= (1 + np.dot(daily_return, self.weights))
            self.weights *= (1 + daily_return)
            self.weights /= np.sum(self.weights)  # Normalize weights after drift
            portfolio_values.append(self.portfolio_value)
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        # print(f"Cumulative Return: {cumulative_return}")

        if not np.isnan(cumulative_return):
            self.portfolio_value *= (1 + cumulative_return)
        else:
            cumulative_return = 0  # Avoid NaN in reward calculation
        reward = np.log(self.portfolio_value / self.initial_balance)

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_state = self.data.iloc[self.current_step].values if not done else np.zeros_like(self.data.iloc[0].values)
        return next_state, reward, done, {}

        def render(self):
            print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}")

    def render(self):
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}")


