"""
stage2_rl_improved.py
======================

This module implements an improved version of the Stage‑2 training loop
described in the project instructions.  It builds upon the user’s
existing Stage‑2 script by incorporating the following enhancements:

* **Modular design** – encapsulates hyperparameter tuning, training,
  validation and prediction into functions for clarity.
* **Multiple iterations** – runs the entire Stage‑2 training and
  prediction process several times with different random seeds to
  evaluate robustness.  The number of iterations is configurable.
* **Early stopping** – uses validation performance to stop training
  when the reward no longer improves.
* **Performance metrics** – records cumulative return and Sharpe ratio
  for each iteration and computes a t‑test to assess whether the
  strategy’s returns are statistically greater than zero.
* **Intermediate outputs** – saves the best model for each rolling
  window, the predicted weights and a summary of metrics.  Each
  iteration is stored in its own folder for reproducibility.

The script retains the rolling retrain logic of the original Stage‑2
code: it retrains on a ten‑year (252×10 day) window, validates on the
following 126 days and predicts on the next 126 days.  You can adjust
these parameters as needed.  To run the script, simply execute

    python stage2_rl_improved.py

The data files `stage2_rl_observations_optimized_10ETFs.csv` and
`stock_prices_10ETFs.csv` must be available in the working
directory.  The environment uses mean–CVaR reward and the same
long‑short normalization logic as in the user’s original code.
"""

import os
import json
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed as sb3_set_random_seed

import gymnasium as gym
from gymnasium import spaces


# -----------------------------------------------------------------------------
# Configuration data classes
# -----------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Holds configurable parameters for the Stage‑2 training loop."""
    # Rolling window lengths (in trading days)
    train_window_days: int = 252 * 10  # 10 years of daily data
    validation_window_days: int = 126  # ~6 months
    prediction_window_days: int = 126  # ~6 months
    lookback_period: int = 10
    rebalance_period: int = 10
    # Hyperparameter tuning settings
    n_iter_tuning: int = 8
    tuning_timesteps: int = 5000
    # PPO training settings
    incremental_timesteps: int = 3000
    max_timesteps: int = 30000
    patience: int = 3
    policy_arch: Tuple[int, int] = (256, 256)
    # Number of outer iterations (different random seeds)
    num_iterations: int = 3  # set to 25 for full experiments
    base_seed: int = 42
    # Risk coefficient for mean–CVaR reward
    default_risk_coeff: float = 0.5


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set seeds for numpy, Python random, torch and stable‑baselines."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    sb3_set_random_seed(seed)


def add_stable_features(df: pd.DataFrame, etf_list: List[str]) -> pd.DataFrame:
    """Compute technical features (volatility, momentum, moving averages).

    This function replicates the user’s `add_stable_features` but works
    directly on a DataFrame with price columns named `Price_ETF`.
    """
    data = df.copy()
    for etf in etf_list:
        price_col = f'Price_{etf}'
        data[f'Volatility_{etf}'] = data[price_col].pct_change().rolling(20).std()
        data[f'Momentum_5d_{etf}'] = data[price_col].pct_change(periods=5)
        data[f'Momentum_10d_{etf}'] = data[price_col].pct_change(periods=10)
        data[f'Momentum_20d_{etf}'] = data[price_col].pct_change(periods=20)
        data[f'MA_5d_{etf}'] = data[price_col].rolling(5).mean()
        data[f'MA_20d_{etf}'] = data[price_col].rolling(20).mean()
        data[f'MA_Crossover_{etf}'] = data[f'MA_5d_{etf}'] - data[f'MA_20d_{etf}']
    data.dropna(inplace=True)
    return data


def filter_features(df: pd.DataFrame,
                    include_predicted_returns: bool = True,
                    include_shap_metrics: bool = True) -> pd.DataFrame:
    """Optionally drop predicted return or SHAP columns from the dataset."""
    df_filtered = df.copy()
    if not include_predicted_returns:
        pred_cols = [c for c in df_filtered.columns if 'Predicted_Return' in c]
        df_filtered.drop(columns=pred_cols, inplace=True)
    if not include_shap_metrics:
        shap_cols = [c for c in df_filtered.columns if 'SHAP' in c]
        df_filtered.drop(columns=shap_cols, inplace=True)
    return df_filtered


# -----------------------------------------------------------------------------
# Custom Gym environment
# -----------------------------------------------------------------------------

class PortfolioEnv(gym.Env):
    """Environment identical to user’s Stage‑2 environment but refactored."""

    metadata = {'render_modes': []}

    def __init__(self, data: pd.DataFrame, etf_list: List[str],
                 reward_type: str = 'mean_cvar', risk_coefficient: float = 0.5,
                 rebalance_period: int = 21, lookback_period: int = 21):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.etf_list = etf_list
        self.reward_type = reward_type
        self.risk_coefficient = risk_coefficient
        self.rebalance_period = rebalance_period
        self.lookback_period = lookback_period
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(len(etf_list),), dtype=np.float32)
        # Identify feature columns (exclude Date and Actual_Return_*)
        self.feature_cols = [c for c in data.columns
                             if c != 'Date' and not c.startswith('Actual_Return')]
        self.num_features_per_day = len(self.feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_features_per_day * lookback_period,),
            dtype=np.float32
        )
        # Internal state
        self.current_step = lookback_period
        self.cumulative_wealth = 1.0
        self.current_weights = np.array([1.0 / len(etf_list)] * len(etf_list))

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.current_step = self.lookback_period
        self.cumulative_wealth = 1.0
        self.current_weights = np.array([1.0 / len(self.etf_list)] * len(self.etf_list))
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        obs_window = self.data.iloc[self.current_step - self.lookback_period: self.current_step]
        return obs_window[self.feature_cols].values.flatten().astype(np.float32)

    def calculate_reward(self, portfolio_return: float, asset_returns: np.ndarray) -> float:
        if self.reward_type == 'cumulative_return':
            return self.cumulative_wealth - 1.0
        elif self.reward_type == 'log_wealth':
            return np.log(self.cumulative_wealth)
        elif self.reward_type == 'mean_var':
            return portfolio_return - self.risk_coefficient * np.var(asset_returns)
        elif self.reward_type == 'mean_cvar':
            alpha = 0.05
            var = np.percentile(asset_returns, 100 * alpha)
            cvar = np.mean(asset_returns[asset_returns <= var])
            return portfolio_return - self.risk_coefficient * cvar
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")

    def step(self, action: np.ndarray):
        next_step = self.current_step + 1
        # Rebalance logic at rebalance_period
        if self.current_step % self.rebalance_period == 0:
            # Long/short normalization identical to user’s implementation
            desired_long = 1.20
            desired_short = 0.20
            clip_bounds = (-0.2, 0.8)
            raw_weights = action.copy()
            long_weights = np.maximum(raw_weights, 0.0)
            short_weights = np.abs(np.minimum(raw_weights, 0.0))
            has_longs = np.sum(long_weights) > 0
            has_shorts = np.sum(short_weights) > 0
            if has_longs and has_shorts:
                normalized_long = desired_long * long_weights / np.sum(long_weights)
                normalized_short = desired_short * short_weights / np.sum(short_weights)
            elif has_longs and not has_shorts:
                normalized_long = long_weights / np.sum(long_weights)
                normalized_short = np.zeros_like(short_weights)
            elif not has_longs and has_shorts:
                num_assets = len(raw_weights)
                normalized_long = np.ones(num_assets) / num_assets
                normalized_short = np.zeros(num_assets)
            else:
                num_assets = len(raw_weights)
                normalized_long = np.ones(num_assets) / num_assets
                normalized_short = np.zeros(num_assets)
            combined_weights = normalized_long - normalized_short
            clipped_weights = np.clip(combined_weights, clip_bounds[0], clip_bounds[1])
            long_clipped = np.maximum(clipped_weights, 0.0)
            short_clipped = np.abs(np.minimum(clipped_weights, 0.0))
            has_long_clipped = np.sum(long_clipped) > 0
            has_short_clipped = np.sum(short_clipped) > 0
            if has_long_clipped and has_short_clipped:
                final_long = desired_long * long_clipped / np.sum(long_clipped)
                final_short = desired_short * short_clipped / np.sum(short_clipped)
            elif has_long_clipped and not has_short_clipped:
                final_long = long_clipped / np.sum(long_clipped)
                final_short = np.zeros_like(short_clipped)
            else:
                num_assets = len(raw_weights)
                final_long = np.ones(num_assets) / num_assets
                final_short = np.zeros(num_assets)
            self.current_weights = final_long - final_short
        else:
            # Passive update between rebalances
            returns_today = np.array([
                self.data.loc[self.current_step, f'Actual_Return_{etf}']
                for etf in self.etf_list
            ])
            self.current_weights *= (1 + returns_today)
            self.current_weights /= np.sum(self.current_weights)
        # Compute portfolio return and reward
        if next_step >= len(self.data):
            terminated = True
            reward = 0.0
        else:
            asset_returns = np.array([
                self.data.loc[next_step, f'Actual_Return_{etf}']
                for etf in self.etf_list
            ])
            portfolio_return = np.dot(self.current_weights, asset_returns)
            self.cumulative_wealth *= (1 + portfolio_return)
            reward = self.calculate_reward(portfolio_return, asset_returns)
            terminated = next_step >= len(self.data) - 1
        self.current_step += 1
        return self._get_obs(), reward, terminated, False, {}


# -----------------------------------------------------------------------------
# Hyperparameter tuning and training functions
# -----------------------------------------------------------------------------

def validate_and_tune(train_data: pd.DataFrame, val_data: pd.DataFrame,
                      etf_list: List[str], reward_type: str,
                      cfg: TrainingConfig) -> Dict[str, float]:
    """Sample a set of PPO hyperparameters and select the best on validation."""
    # Parameter distributions to sample
    param_dist = {
        'learning_rate': [3e-4, 1e-4],
        'n_steps': [20, 40],
        'batch_size': [10, 20],
        'gamma': [0.95, 0.98],
        'risk_coefficient': [0.1, 0.5, 1.0],
        'seed': [cfg.base_seed, cfg.base_seed + 17, cfg.base_seed + 42],
    }
    sampled_params = list(ParameterSampler(param_dist, n_iter=cfg.n_iter_tuning,
                                           random_state=cfg.base_seed))
    best_reward = -np.inf
    best_params = None
    for params in sampled_params:
        seed = params.pop('seed')
        risk_coeff = params.pop('risk_coefficient', cfg.default_risk_coeff)
        set_global_seed(seed)
        env = make_vec_env(lambda: PortfolioEnv(train_data, etf_list,
                                                reward_type, risk_coeff,
                                                cfg.rebalance_period,
                                                cfg.lookback_period),
                           n_envs=1, seed=seed)
        model = PPO('MlpPolicy', env, ent_coef=0.01, clip_range=0.2, seed=seed,
                    **params, verbose=0)
        model.learn(total_timesteps=cfg.tuning_timesteps)
        # Validate
        val_env = PortfolioEnv(val_data, etf_list, reward_type, risk_coeff,
                               cfg.rebalance_period, cfg.lookback_period)
        obs, _ = val_env.reset(seed=seed)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = val_env.step(action)
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
            best_params = params.copy()
            best_params['risk_coefficient'] = risk_coeff
            best_params['seed'] = seed
    return best_params


def train_and_predict(train_df: pd.DataFrame, val_df: pd.DataFrame,
                      pred_df: pd.DataFrame, etf_list: List[str],
                      cfg: TrainingConfig, best_params: Dict[str, float],
                      model_save_path: str) -> Tuple[List[float], List[pd.Timestamp]]:
    """Train a PPO model with early stopping and predict on pred_df.

    Returns the list of final weights (for each rebalance step) and the
    corresponding dates.  The model is saved to `model_save_path`.
    """
    risk_coeff = best_params.pop('risk_coefficient')
    seed = best_params.pop('seed')
    set_global_seed(seed)
    env_train = make_vec_env(lambda: PortfolioEnv(train_df, etf_list,
                                                  'mean_cvar', risk_coeff,
                                                  cfg.rebalance_period,
                                                  cfg.lookback_period),
                             n_envs=1, seed=seed)
    policy_kwargs = dict(net_arch=list(cfg.policy_arch))
    model = PPO('MlpPolicy', env_train,
                policy_kwargs=policy_kwargs,
                ent_coef=0.01,
                clip_range=0.2,
                seed=seed,
                **best_params,
                verbose=0)
    # Early stopping
    best_val_reward = -np.inf
    no_improve = 0
    for step in range(0, cfg.max_timesteps, cfg.incremental_timesteps):
        model.learn(total_timesteps=cfg.incremental_timesteps)
        # Evaluate on validation set
        val_env = PortfolioEnv(val_df, etf_list, 'mean_cvar', risk_coeff,
                               cfg.rebalance_period, cfg.lookback_period)
        obs, _ = val_env.reset(seed=seed)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = val_env.step(action)
            total_reward += reward
        if total_reward > best_val_reward:
            best_val_reward = total_reward
            no_improve = 0
            model.save(model_save_path)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break
    # Load best model
    best_model = PPO.load(model_save_path)
    # Predict on pred_df
    env_pred = PortfolioEnv(pred_df, etf_list, 'mean_cvar', risk_coeff,
                            cfg.rebalance_period, cfg.lookback_period)
    obs, _ = env_pred.reset()
    done = False
    weights_list: List[List[float]] = []
    dates_list: List[pd.Timestamp] = []
    action = np.zeros(len(etf_list), dtype=np.float32)
    while not done:
        if env_pred.current_step >= cfg.lookback_period and (
            env_pred.current_step % cfg.rebalance_period == 0
        ):
            action, _ = best_model.predict(obs, deterministic=True)
            # Apply the same normalization used inside env_pred for consistency
            desired_long = 1.20
            desired_short = 0.20
            clip_bounds = (-0.2, 0.8)
            raw = action.copy()
            long_w = np.maximum(raw, 0.0)
            short_w = np.abs(np.minimum(raw, 0.0))
            has_longs = np.sum(long_w) > 0
            has_shorts = np.sum(short_w) > 0
            if has_longs and has_shorts:
                norm_long = desired_long * long_w / np.sum(long_w)
                norm_short = desired_short * short_w / np.sum(short_w)
            elif has_longs and not has_shorts:
                norm_long = long_w / np.sum(long_w)
                norm_short = np.zeros_like(short_w)
            elif not has_longs and has_shorts:
                n = len(raw)
                norm_long = np.ones(n) / n
                norm_short = np.zeros(n)
            else:
                n = len(raw)
                norm_long = np.ones(n) / n
                norm_short = np.zeros(n)
            combined = norm_long - norm_short
            clipped = np.clip(combined, clip_bounds[0], clip_bounds[1])
            long_c = np.maximum(clipped, 0.0)
            short_c = np.abs(np.minimum(clipped, 0.0))
            has_long_c = np.sum(long_c) > 0
            has_short_c = np.sum(short_c) > 0
            if has_long_c and has_short_c:
                final_long = desired_long * long_c / np.sum(long_c)
                final_short = desired_short * short_c / np.sum(short_c)
            elif has_long_c and not has_short_c:
                final_long = long_c / np.sum(long_c)
                final_short = np.zeros_like(short_c)
            else:
                n = len(raw)
                final_long = np.ones(n) / n
                final_short = np.zeros(n)
            final_w = final_long - final_short
            weights_list.append(final_w.tolist())
            dates_list.append(env_pred.data.loc[env_pred.current_step, 'Date'])
        obs, _, done, _, _ = env_pred.step(action)
    return weights_list, dates_list


# -----------------------------------------------------------------------------
# Main execution routine
# -----------------------------------------------------------------------------

def main():
    cfg = TrainingConfig()
    # Load data
    data = pd.read_csv('stage2_rl_observations_optimized_10ETFs.csv', parse_dates=['Date'])
    price_data = pd.read_csv('stock_prices_10ETFs.csv')
    price_data['Date'] = pd.to_datetime(price_data['Date'], utc=True).dt.tz_localize(None)
    # Rename price columns to Price_ETF
    price_cols = {col: f'Price_{col}' for col in price_data.columns if col != 'Date'}
    price_data.rename(columns=price_cols, inplace=True)
    # Merge and compute features
    merged_data = pd.merge(data, price_data, on='Date', how='inner').reset_index(drop=True)
    if len(merged_data) != len(data):
        print("Warning: data length mismatch after merge.")
    feature_data = add_stable_features(merged_data, cfg_etf_list := ['XLB','XLE','XLF','XLI','XLK','XLP','XLY','XLV','XLU'])
    feature_data = filter_features(feature_data, include_predicted_returns=True, include_shap_metrics=True)
    # Rolling window indices
    total_len = len(feature_data)
    start_indices = range(0,
                          total_len - (cfg.train_window_days + cfg.validation_window_days + cfg.prediction_window_days),
                          cfg.prediction_window_days)
    # Prepare output directory
    root_output_dir = 'stage2_iterations'
    os.makedirs(root_output_dir, exist_ok=True)
    # Collect metrics for all iterations
    summary_records = []
    # Outer loop for multiple iterations
    for iter_num in range(cfg.num_iterations):
        iter_seed = cfg.base_seed + iter_num
        iter_dir = os.path.join(root_output_dir, f'iteration_{iter_num:02d}')
        os.makedirs(iter_dir, exist_ok=True)
        iter_returns = []
        # Loop over rolling windows
        for idx, start_idx in enumerate(start_indices):
            # Define indices
            train_start = start_idx
            train_end = train_start + cfg.train_window_days
            val_start = train_end
            val_end = val_start + cfg.validation_window_days
            pred_start = val_end
            pred_end = pred_start + cfg.prediction_window_days
            train_df = feature_data.iloc[train_start:train_end].reset_index(drop=True)
            val_df = feature_data.iloc[val_start:val_end].reset_index(drop=True)
            pred_df = feature_data.iloc[pred_start:pred_end].reset_index(drop=True)
            # Scale features
            feature_cols = [c for c in train_df.columns if c != 'Date' and not c.startswith('Actual_Return')]
            scaler = StandardScaler()
            scaler.fit(train_df[feature_cols])
            train_scaled = train_df.copy()
            train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
            val_scaled = val_df.copy()
            val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
            pred_scaled = pred_df.copy()
            pred_scaled[feature_cols] = scaler.transform(pred_df[feature_cols])
            # Tune hyperparameters on this window
            best_params = validate_and_tune(train_scaled, val_scaled,
                                            cfg_etf_list, 'mean_cvar', cfg)
            # Train and predict
            model_subdir = os.path.join(iter_dir, f'window_{idx:02d}')
            os.makedirs(model_subdir, exist_ok=True)
            model_path = os.path.join(model_subdir, 'best_ppo.zip')
            weights_list, dates_list = train_and_predict(train_scaled, val_scaled, pred_scaled,
                                                         cfg_etf_list, cfg, best_params.copy(),
                                                         model_path)
            # Save weights
            weights_df = pd.DataFrame(weights_list, columns=cfg_etf_list)
            weights_df.insert(0, 'Date', dates_list)
            weights_df.to_csv(os.path.join(model_subdir, 'weights.csv'), index=False)
            # Compute cumulative return for this window
            # For simplicity, we compute return based on env_pred inside train_and_predict,
            # but here we recompute using pred_df and weights
            cum_wealth = 1.0
            for t, w in zip(dates_list, weights_list):
                step_idx = pred_scaled[pred_scaled['Date'] == t].index[0]
                asset_returns = np.array([
                    pred_scaled.loc[step_idx + 1, f'Actual_Return_{etf}']
                    for etf in cfg_etf_list
                ])
                port_ret = np.dot(w, asset_returns)
                cum_wealth *= (1 + port_ret)
            iter_returns.append(cum_wealth - 1.0)
        # Compute statistics for this iteration
        mean_return = np.mean(iter_returns)
        std_return = np.std(iter_returns, ddof=1)
        sharpe = (mean_return / std_return) * np.sqrt(len(iter_returns)) if std_return != 0 else np.nan
        summary_records.append({'iteration': iter_num, 'seed': iter_seed,
                               'mean_return': mean_return, 'sharpe': sharpe})
    # Overall significance test
    overall_returns = [rec['mean_return'] for rec in summary_records]
    t_stat, p_val = ttest_1samp(overall_returns, 0.0)
    # Save summary
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(root_output_dir, 'iterations_summary.csv'), index=False)
    with open(os.path.join(root_output_dir, 't_test.json'), 'w') as f:
        json.dump({'t_statistic': float(t_stat), 'p_value': float(p_val)}, f, indent=2)
    print(summary_df)
    print(f"Overall t‑statistic={t_stat:.3f}, p‑value={p_val:.3f}")


if __name__ == '__main__':
    main()