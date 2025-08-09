import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
shap_metrics = pd.read_csv('shap_value_metrics_export.csv', parse_dates=['Start_Date', 'End_Date'])
factor_returns = pd.read_csv('aligned_factors.csv', parse_dates=['Date'], index_col='Date')
stock_returns = pd.read_csv('daily_returns.csv', parse_dates=['Date'], index_col='Date')

# Filter test phase SHAP metrics
shap_metrics_test = shap_metrics[shap_metrics['Phase'] == 'Test'].copy()

selected_columns = [
    'mean_abs_shap_Mkt-RF', 'mean_abs_shap_SMB', 'mean_abs_shap_HML',
    'mean_abs_shap_RMW', 'mean_abs_shap_CMA',
    'shap_std_Mkt-RF', 'shap_std_SMB', 'shap_std_HML',
    'shap_std_RMW', 'shap_std_CMA',
    'mean_abs_over_std_Mkt-RF', 'mean_abs_over_std_SMB', 'mean_abs_over_std_HML',
    'mean_abs_over_std_RMW', 'mean_abs_over_std_CMA'
]

# Aggregate by (Start_Date, End_Date)
shap_agg = shap_metrics_test.groupby(['Start_Date', 'End_Date'])[selected_columns].mean().reset_index()

# Standardize SHAP metrics
scaler_shap = StandardScaler()
shap_agg[selected_columns] = scaler_shap.fit_transform(shap_agg[selected_columns])

daily_regimes = pd.DataFrame(index=stock_returns.index)

for _, row in shap_agg.iterrows():
    mask = (daily_regimes.index >= row['Start_Date']) & (daily_regimes.index <= row['End_Date'])
    daily_regimes.loc[mask, selected_columns] = row[selected_columns].values

daily_regimes.ffill(inplace=True)
daily_regimes.dropna(inplace=True)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
daily_regimes['Cluster'] = kmeans.fit_predict(daily_regimes[selected_columns])

factor_scaler = StandardScaler()
factor_returns_norm = pd.DataFrame(
    factor_scaler.fit_transform(factor_returns),
    index=factor_returns.index,
    columns=factor_returns.columns
)

stock_scaler = StandardScaler()
stock_returns_norm = pd.DataFrame(
    stock_scaler.fit_transform(stock_returns),
    index=stock_returns.index,
    columns=stock_returns.columns
)

merged_data = stock_returns_norm.join(daily_regimes[['Cluster'] + selected_columns], how='inner') \
                                .join(factor_returns_norm, rsuffix='_factor') \
                                .dropna()

merged_data.to_csv("final_merged_data.csv")

regime_series = daily_regimes['Cluster'].copy()
K = regime_series.nunique()

transition_matrix = np.zeros((K, K))
for (prev, curr) in zip(regime_series[:-1], regime_series[1:]):
    transition_matrix[prev, curr] += 1

transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

w = 10  # window size
forecast_dates, forecast_probs = [], []

for i in range(w, len(regime_series)):
    recent = regime_series.iloc[i - w:i]
    pt = np.array([(recent == k).mean() for k in range(K)])
    pt1 = pt @ transition_matrix
    forecast_dates.append(regime_series.index[i])
    forecast_probs.append(pt1)

forecast_df = pd.DataFrame(
    forecast_probs, 
    columns=[f'P_Regime_{k}' for k in range(K)], 
    index=forecast_dates
)

forecast_df.to_csv("forecasted_regime_probabilities.csv")

merged_data_with_forecast = merged_data.join(forecast_df, how='left').fillna(method='ffill')
merged_data_with_forecast.to_csv("final_merged_data_with_forecast.csv")


#############################################################
### Insert the following code snippet after you have created the file final_merged_data_with_forecast.csv:

# Incorporates the forecasted regime probabilities (P_Regime_0, P_Regime_1) into the high-level PPO’s state space.

# Retains the hierarchical setup:

    # High-level PPO dynamically picks the market regime.

    # Low-level PPO optimizes allocations conditioned on chosen regime.

# Outputs clearly structured results (hierarchical_allocations_dates_rebalanced.csv) with cumulative returns and portfolio allocations per window and regime.

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# === Load your enhanced merged dataset ===
merged_data = pd.read_csv("final_merged_data_with_forecast.csv", parse_dates=['Date'], index_col='Date')

# Load aligned raw returns for accurate evaluation
raw_stock_returns = pd.read_csv("aligned_raw_stock_returns.csv", parse_dates=['Date'], index_col='Date')

# Define important variables
train_window, test_window = 252, 63
rebalance_freq = 21
model_save_dir = './trained_hierarchical_model'
os.makedirs(model_save_dir, exist_ok=True)

# Define columns explicitly
tickers = [col for col in raw_stock_returns.columns]
forecast_cols = [col for col in merged_data.columns if col.startswith('P_Regime_')]
regime_cols = ['Cluster'] + forecast_cols
factor_cols = [col for col in merged_data.columns if col.endswith('_factor')]

total_windows = len(range(0, len(merged_data) - train_window - test_window, test_window))
records = []

# Start hierarchical training/testing loop
for i, start in enumerate(range(0, len(merged_data) - train_window - test_window, test_window)):
    print(f"\n=== Window {i+1}/{total_windows} ===")

    # === Split train and test explicitly ===
    train_df = merged_data.iloc[start:start + train_window]
    test_df = merged_data.iloc[start + train_window:start + train_window + test_window + 1]
    raw_returns_train = raw_stock_returns.loc[train_df.index]
    raw_returns_test = raw_stock_returns.loc[test_df.index]

    # === High-Level Training Data Preparation ===
    high_features_train = train_df[regime_cols + factor_cols][:-rebalance_freq]
    future_returns_high = np.array([
        raw_returns_train.iloc[idx + 1: idx + rebalance_freq + 1].mean().mean()
        for idx in range(len(high_features_train))
    ])

    # Environment for high-level agent (Regime Selector)
    high_env = DummyVecEnv([lambda: RegimeSelectorEnv(high_features_train, future_returns_high)])

    high_model_path = os.path.join(model_save_dir, f'high_model_window_{i-1}.zip')
    new_high_model_path = os.path.join(model_save_dir, f'high_model_window_{i}.zip')

    if i > 0 and os.path.exists(high_model_path):
        high_model = PPO.load(high_model_path, env=high_env)
        high_model.learn(total_timesteps=20000)
    else:
        high_model = PPO('MlpPolicy', high_env, verbose=0)
        high_model.learn(total_timesteps=10000)

    high_model.save(new_high_model_path)

    # === Generate historical regime predictions explicitly ===
    historical_regimes = []
    for obs in high_features_train.values:
        regime, _ = high_model.predict(obs, deterministic=True)
        historical_regimes.append(int(regime))

    historical_regimes = pd.Series(historical_regimes, index=high_features_train.index)

    # === Low-Level Training ===
    low_level_models = {}
    for regime in historical_regimes.unique():
        regime_indices = historical_regimes[historical_regimes == regime].index
        regime_df = train_df.loc[regime_indices]
        chosen_regime_vector_train = np.repeat(regime, len(regime_df))

        low_env_train = DummyVecEnv([
            lambda: RegimeConditionedPortfolioEnv(
                returns_df=regime_df[tickers],
                chosen_regime_vector=chosen_regime_vector_train,
                raw_returns_df=raw_returns_train.loc[regime_df.index]
            )
        ])

        prev_low_model_path = os.path.join(model_save_dir, f'low_model_window_{i-1}_regime_{regime}.zip')
        current_low_model_path = os.path.join(model_save_dir, f'low_model_window_{i}_regime_{regime}.zip')

        if i > 0 and os.path.exists(prev_low_model_path):
            low_model = PPO.load(prev_low_model_path, env=low_env_train)
            low_model.learn(total_timesteps=20000)
        else:
            low_model = PPO('MlpPolicy', low_env_train, verbose=0)
            low_model.learn(total_timesteps=10000)

        low_model.save(current_low_model_path)
        low_level_models[regime] = low_model

    # === Testing ===
    test_returns = test_df[tickers].values
    test_regimes = test_df[regime_cols + factor_cols].values
    test_dates = test_df.index

    for rebalance_num, rebalance_start in enumerate(range(0, test_window, rebalance_freq)):
        rebalance_end = min(rebalance_start + rebalance_freq, test_window)

        high_obs = test_regimes[rebalance_start]
        chosen_regime, _ = high_model.predict(high_obs, deterministic=True)

        chosen_regime_vector_test = np.repeat(chosen_regime, rebalance_end - rebalance_start)

        if chosen_regime in low_level_models:
            low_model = low_level_models[chosen_regime]
            low_env_test = RegimeConditionedPortfolioEnv(
                returns_df=pd.DataFrame(test_returns[rebalance_start:rebalance_end], columns=tickers),
                chosen_regime_vector=chosen_regime_vector_test,
                raw_returns_df=raw_returns_test.iloc[rebalance_start:rebalance_end]
            )

            low_obs = low_env_test.reset()
            action, _ = low_model.predict(low_obs, deterministic=True)
            action /= action.sum()
        else:
            action = np.ones(len(tickers)) / len(tickers)

        # Evaluate portfolio performance
        portfolio_value = 1.0
        for step in range(rebalance_end - rebalance_start - 1):
            reward = np.dot(action, raw_returns_test.iloc[rebalance_start + step + 1])
            portfolio_value *= (1 + reward)
        cumulative_reward = portfolio_value - 1.0

        weights_dict = {ticker: round(weight, 4) for ticker, weight in zip(tickers, action)}
        records.append({
            'Rebalance_Date': test_dates[rebalance_start],
            'Window': i,
            'Rebalance_Number': rebalance_num,
            'Chosen_Regime': int(chosen_regime),
            'Holding_Period_Start': test_dates[rebalance_start],
            'Holding_Period_End': test_dates[rebalance_end - 1],
            'Cumulative_Reward': cumulative_reward,
            **weights_dict
        })

result_df = pd.DataFrame(records)
result_df['Cumulative_Return'] = result_df['Cumulative_Reward'].cumsum()
result_df.to_csv('hierarchical_allocations_dates_rebalanced.csv', index=False)

print("\nHierarchical training/testing completed successfully.")
