# Complete attention model pipeline explicitly matching XGBoost-generated results format
import pandas as pd
import numpy as np
import shap
import ta
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
factors = pd.read_csv("aligned_factors.csv", index_col=0, parse_dates=True)
returns = pd.read_csv("daily_returns_10ETFs.csv", index_col=0, parse_dates=True)

# Align dates
dates = factors.index.intersection(returns.index)
factors = factors.loc[dates]
returns = returns.loc[dates]

# Compute technical indicators
all_tech_features = []
for etf in returns.columns:
    close = (1 + returns[etf]).cumprod()
    etf_tech_features = pd.DataFrame(index=returns.index)

    etf_tech_features[f'{etf}_SMA_5'] = ta.trend.sma_indicator(close, 5)
    etf_tech_features[f'{etf}_SMA_20'] = ta.trend.sma_indicator(close, 20)
    etf_tech_features[f'{etf}_RSI_14'] = ta.momentum.rsi(close, 14)
    etf_tech_features[f'{etf}_MACD'] = ta.trend.macd_diff(close)
    etf_tech_features[f'{etf}_Volatility_20'] = returns[etf].rolling(20).std()
    etf_tech_features[f'{etf}_Momentum_10'] = returns[etf].rolling(10).mean()

    all_tech_features.append(etf_tech_features)

technical_features = pd.concat(all_tech_features, axis=1)
features = pd.concat([factors, technical_features], axis=1).dropna()
target_returns = returns.shift(-1).loc[features.index].dropna()
features = features.loc[target_returns.index]

# Attention model definition
class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=2, batch_first=True)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.fc(attn_output[:, -1, :]).squeeze()

# Helper functions
def tensorize(X):
    return torch.tensor(X.values, dtype=torch.float32).unsqueeze(0)

# Rolling window training and prediction
results = []
train_years, valid_years, test_years = 10, 2, 1
start_year, end_year = 2009, 2024

for etf in returns.columns:
    year = start_year
    while year <= end_year - test_years + 1:
        train_start, train_end = f'{year - train_years}-01-01', f'{year - valid_years - 1}-12-31'
        valid_start, valid_end = f'{year - valid_years}-01-01', f'{year - 1}-12-31'
        test_start, test_end = f'{year}-01-01', f'{year + test_years - 1}-12-31'

        X_train, y_train = features.loc[train_start:train_end], target_returns[etf].loc[train_start:train_end]
        X_valid, y_valid = features.loc[valid_start:valid_end], target_returns[etf].loc[valid_start:valid_end]
        X_test, y_test = features.loc[test_start:test_end], target_returns[etf].loc[test_start:test_end]

        model = AttentionModel(input_dim=X_train.shape[1], hidden_dim=64)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(tensorize(X_train))
            loss = criterion(output, torch.tensor(y_train.values, dtype=torch.float32))
            loss.backward()
            optimizer.step()

        # Predict
        model.eval()
        with torch.no_grad():
            preds = model(tensorize(X_test)).numpy()

        # SHAP values
        explainer = shap.DeepExplainer(model, tensorize(X_train))
        shap_values = explainer.shap_values(tensorize(X_test))[0]

        shap_df = pd.DataFrame(shap_values, columns=[f'SHAP_{col}' for col in X_test.columns], index=X_test.index).reset_index()
        shap_df.rename(columns={'index': 'Date'}, inplace=True)

        preds_df = pd.DataFrame({
            'Date': X_test.index,
            'ETF': etf,
            'Year': year,
            'Actual_Return': y_test.values,
            'Predicted_Return': preds
        }).reset_index(drop=True)

        combined_df = preds_df.merge(shap_df, on='Date', how='left')
        results.append(combined_df)

        year += 1

final_results = pd.concat(results).reset_index(drop=True)
final_results.to_csv("attention_stage1_predictions_with_shap.csv", index=False)
