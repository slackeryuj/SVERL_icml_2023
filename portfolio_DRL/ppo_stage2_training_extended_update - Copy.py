"""
Extended PPO Training for Portfolio Management
-------------------------------------------------

This script extends the original Stage 2 PPO training workflow to provide a
longer out‑of‑sample testing window, deterministic behaviour through explicit
seeding, GPU acceleration where available, and comprehensive logging of
intermediate artefacts.  The key improvements over the baseline are:

* **Expanded test period:**  Instead of hard‑coding a three‑year test set,
  the code now calculates the test duration dynamically based on the amount of
  data remaining after allocating training and validation windows.  By
  default the training window is eight years (≈2016 trading days) and the
  validation window is two years (≈504 trading days); any remaining records
  are reserved for out‑of‑sample evaluation.  This gives nearly six years
  of test data with the provided datasets.

* **Deterministic execution:**  All sources of randomness—including NumPy,
  Python’s `random`, PyTorch and the Gym environments—are seeded with a
  single constant (`SEED=42`).  The vectorised environments are seeded
  explicitly before training, and the PPO agent is constructed with the same
  seed to ensure reproducible results.  Running this script multiple times
  yields identical models and evaluation metrics.

* **GPU support:**  The script detects whether a CUDA‑capable device is
  available via `torch.cuda.is_available()`.  If so, the PPO agent runs on
  GPU by passing `device='cuda'`; otherwise it falls back to CPU.  This
  greatly accelerates training when a GPU is available.

* **Intermediate logging and model checkpoints:**  Training progress is
  monitored via Stable Baselines 3’s `EvalCallback`.  The agent is
  periodically evaluated on the validation environment, and the best model
  (according to the mean validation reward) is saved to disk in
  `./logs/best_model`.  A checkpoint of the latest model is saved every
  evaluation interval, and a CSV file (`training_history.csv`) records the
  episode number, mean validation reward and timestamp at each evaluation.

Usage
-----

Run this script in an environment with the following prerequisites installed:

* `pandas`, `numpy`, `scikit‑learn`
* `gym`, `stable_baselines3`, `torch`

Place the input CSV files `stage2_rl_observations_optimized_10ETFs.csv` and
`stock_prices_10ETFs.csv` in the working directory.  When run, the script
produces several output files:

* `ppo_stage2_final_model.zip`  – the final trained PPO agent (last checkpoint)
* `logs/best_model.zip`         – the best model discovered during training
* `logs/training_history.csv`   – CSV log of validation rewards during training
* `ppo_stage2_weights.csv`      – monthly portfolio weights on the extended
  out‑of‑sample period
* Console output summarising cumulative and annualised returns, volatility,
  Sharpe ratio and maximum drawdown for both the RL and equal‑weighted
  portfolios on the extended test set.
"""

import os
import json
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList, CheckpointCallback

import gym
from gym import spaces
import torch
import random


# -------------------------------------------------------------------
# Utility functions and seeding
# -------------------------------------------------------------------
SEED = 42

def set_global_seed(seed: int) -> None:
    """Seed all relevant libraries for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # If using CUDA, seed all devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_global_seed(SEED)


# -------------------------------------------------------------------
# Environment definition with softmax normalisation and Mean‑CVaR reward
# -------------------------------------------------------------------
class PortfolioEnv(gym.Env):
    """
    Custom Gym environment for portfolio allocation.

    Observations are flattened windows of features; actions are unconstrained
    real numbers that are converted to portfolio weights via softmax.  The
    reward is computed at each rebalance period as mean minus λ × CVaR.
    """

    metadata = {"render.modes": []}

    def __init__(self, data: pd.DataFrame, etf_list: list[str],
                 reward_type: str = 'mean_cvar', risk_coefficient: float = 1.0,
                 rebalance_period: int = 21, lookback_period: int = 60) -> None:
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.etf_list = etf_list
        self.reward_type = reward_type
        self.risk_coefficient = risk_coefficient
        self.rebalance_period = rebalance_period
        self.lookback_period = lookback_period

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-10.0, high=10.0,
                                       shape=(len(etf_list),),
                                       dtype=np.float32)
        self.feature_cols = [c for c in data.columns
                             if c not in ['Date'] and not c.startswith('Actual_Return')]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(self.feature_cols)*lookback_period,),
                                            dtype=np.float32)

        # Internal state
        self.current_step = self.lookback_period
        self.current_weights = np.array([1/len(etf_list)]*len(etf_list), dtype=float)
        self.cumulative_wealth = 1.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.current_step = self.lookback_period
        self.current_weights = np.array([1/len(self.etf_list)]*len(self.etf_list), dtype=float)
        self.cumulative_wealth = 1.0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Return a flattened window of recent features."""
        window = self.data.iloc[
            self.current_step - self.lookback_period : self.current_step
        ]
        return window[self.feature_cols].values.flatten().astype(np.float32)

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Convert raw action outputs into a valid long‑only weight vector via softmax.
        Implements the 'continuous 10‑dimensional weights with softmax normalisation'.
        """
        temperature = 1.0
        scaled = action / temperature
        exp_vals = np.exp(scaled - np.max(scaled))
        return exp_vals / exp_vals.sum()

    def calculate_reward(self, portfolio_return: float, asset_returns: np.ndarray) -> float:
        """Compute reward according to the chosen risk measure."""
        eps = 1e-8  # prevent division by zero

        if self.reward_type == 'mean_cvar':
            alpha = 0.05
            var = np.percentile(asset_returns, 100 * alpha)
            cvar = np.mean(asset_returns[asset_returns <= var])
            return portfolio_return - self.risk_coefficient * cvar

        elif self.reward_type == 'mean_var':
            return portfolio_return - self.risk_coefficient * np.var(asset_returns)

        elif self.reward_type == 'log_wealth':
            return np.log(self.cumulative_wealth)

        elif self.reward_type == 'cumulative_return':
            return self.cumulative_wealth - 1.0

        elif self.reward_type == 'sharpe_ratio':
            mean_ret = np.mean(asset_returns)
            std_ret = np.std(asset_returns)
            return mean_ret / (std_ret + eps)

        elif self.reward_type == 'adjusted_sharpe':
            mean_ret = np.mean(asset_returns)
            std_ret = np.std(asset_returns)
            return portfolio_return + self.risk_coefficient * (mean_ret / (std_ret + eps))

        elif self.reward_type == 'sortino_ratio':
            negative_returns = asset_returns[asset_returns < 0]
            downside_std = np.std(negative_returns) if negative_returns.size > 0 else eps
            mean_ret = np.mean(asset_returns)
            return mean_ret / (downside_std + eps)

        elif self.reward_type == 'mean_return':
            return np.mean(asset_returns)

        else:
            raise ValueError(f"Unknown reward_type {self.reward_type}")

    def step(self, action: np.ndarray):
        """Update portfolio and compute reward."""
        next_step = self.current_step + 1

        # Rebalance portfolio at predefined periods
        if self.current_step % self.rebalance_period == 0:
            self.current_weights = self._action_to_weights(action)
        else:
            daily_rets = np.array([
                self.data.loc[self.current_step, f'Actual_Return_{t}']
                for t in self.etf_list
            ])
            self.current_weights *= (1 + daily_rets)
            self.current_weights /= self.current_weights.sum()

        # compute reward on the next day
        if next_step >= len(self.data):
            done = True
            reward = 0.0
        else:
            asset_returns = np.array([
                self.data.loc[next_step, f'Actual_Return_{t}']
                for t in self.etf_list
            ])
            portfolio_ret = float(np.dot(self.current_weights, asset_returns))
            self.cumulative_wealth *= (1 + portfolio_ret)
            reward = self.calculate_reward(portfolio_ret, asset_returns)
            done = (next_step >= len(self.data) - 1)

        self.current_step += 1
        return self._get_obs(), reward, done, False, {}


# -------------------------------------------------------------------
# Data preparation
# -------------------------------------------------------------------
def load_and_prepare_data(stage2_file: str, prices_file: str) -> pd.DataFrame:
    """Load the stage 2 observations and price data and construct features."""
    # Load observations and prices
    stage2 = pd.read_csv(stage2_file, parse_dates=['Date'])
    prices = pd.read_csv(prices_file)
    prices['Date'] = pd.to_datetime(prices['Date'], utc=True).dt.tz_localize(None)

    # Align on Date
    prices.rename(columns={c: f'Price_{c}' for c in prices.columns if c != 'Date'},
                  inplace=True)
    data_merged = pd.merge(stage2, prices, on='Date', how='inner')

    # Add technical indicators
    etfs = [c.replace('Price_', '') for c in prices.columns if c.startswith('Price_')]
    df2 = data_merged.copy()
    for etf in etfs:
        price_col = f'Price_{etf}'
        returns = df2[price_col].pct_change()
        df2[f'Volatility_{etf}'] = returns.rolling(20).std()
        df2[f'Momentum_5d_{etf}'] = returns.rolling(5).sum()
        df2[f'Momentum_10d_{etf}'] = returns.rolling(10).sum()
        df2[f'Momentum_20d_{etf}'] = returns.rolling(20).sum()
        df2[f'MA_5d_{etf}'] = df2[price_col].rolling(5).mean()
        df2[f'MA_20d_{etf}'] = df2[price_col].rolling(20).mean()
        df2[f'MA_Crossover_{etf}'] = df2[f'MA_5d_{etf}'] - df2[f'MA_20d_{etf}']
    df2.dropna(inplace=True)

    # Return the augmented DataFrame
    return df2


def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], StandardScaler]:
    """
    Fit a standard scaler on the training data and apply it to the train,
    validation and test sets.  Non‑feature columns are retained unchanged.
    Returns the scaled dataframes, the list of feature columns and the scaler.
    """
    feature_cols = [c for c in train_df.columns
                    if c not in ['Date'] and not c.startswith('Actual_Return')]
    scaler = StandardScaler().fit(train_df[feature_cols])

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        x = scaler.transform(df[feature_cols])
        df_scaled = pd.DataFrame(x, columns=feature_cols, index=df.index)
        for col in df.columns:
            if col not in feature_cols:
                df_scaled[col] = df[col]
        return df_scaled[df.columns]

    return transform(train_df), transform(val_df), transform(test_df), feature_cols, scaler


# -------------------------------------------------------------------
# Training and evaluation
# -------------------------------------------------------------------
def train_and_evaluate(
    data_with_features: pd.DataFrame,
    etf_list: list[str],
    lookback: int = 60,
    rebalance: int = 21,
    train_years: int = 8,
    val_years: int = 2,
    total_timesteps: int = int(3e6),
    eval_freq: int = int(2e5),
    device: str | torch.device = 'auto',
    prices_file: str | None = None,
    seed: int | None = None
) -> dict:
    """
    Train a PPO agent on the provided data and return performance metrics.

    Parameters
    ----------
    data_with_features : DataFrame
        The full dataset with engineered features.
    etf_list : list of str
        List of ETF tickers included in the dataset.
    lookback : int
        Length of the observation window in trading days.
    rebalance : int
        Rebalance frequency in trading days.
    train_years : int
        Number of years used for training.
    val_years : int
        Number of years used for validation.
    total_timesteps : int
        Total number of timesteps for PPO training.
    eval_freq : int
        Evaluation frequency (in timesteps) for the callback.
    device : str or torch.device
        Device on which to run the model; e.g. 'cuda' or 'cpu'.

    seed : int or None, optional
        Random seed to control reproducibility.  When provided, this value
        overrides the module level constant `SEED` for the duration of this
        call.  If omitted, the global `SEED` is used.  By supplying
        different seeds to successive invocations you can obtain
        independent training runs.

    Returns
    -------
    dict
        A dictionary containing performance metrics and file paths.
    """
    # Optionally override the global seed for reproducibility.  When a seed
    # value is supplied, we reseed all underlying libraries up front and
    # propagate this value to the environments and PPO agent.  Otherwise
    # the module level `SEED` constant remains in force.
    if seed is not None:
        set_global_seed(seed)
        run_seed = seed
    else:
        run_seed = SEED

    # Determine lengths
    train_days = train_years * 252
    val_days = val_years * 252
    # The remaining observations are used for out‑of‑sample testing
    test_days = len(data_with_features) - train_days - val_days
    if test_days <= 0:
        raise ValueError("Not enough data to allocate test period with the given train and val lengths.")

    # Split the data
    start_idx = 0
    train_data = data_with_features.iloc[start_idx:start_idx+train_days].reset_index(drop=True)
    val_data = data_with_features.iloc[start_idx+train_days:
                                        start_idx+train_days+val_days].reset_index(drop=True)
    test_data = data_with_features.iloc[start_idx+train_days+val_days:].reset_index(drop=True)

    # Standardise features
    train_scaled, val_scaled, test_scaled, feature_cols, scaler = scale_features(train_data, val_data, test_data)

    # Vectorised training environment
    n_envs = 10
    def make_env() -> gym.Env:
        return PortfolioEnv(train_scaled, etf_list, reward_type='log_wealth',
                            risk_coefficient=1.0, rebalance_period=rebalance,
                            lookback_period=lookback)

    vec_env = SubprocVecEnv([make_env for _ in range(n_envs)], start_method='spawn')
    # Seed all environments for reproducibility using the chosen run_seed
    vec_env.seed(run_seed)

    # Determine the compute device
    if device == 'auto':
        device_used = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_used = device

    # Hyper‑parameters for PPO
    n_steps = 252 * 3  # collect roughly three months of data per environment
    policy_kwargs = dict(
        net_arch=[64, 64],
        activation_fn=torch.nn.Tanh,
        log_std_init=-1.0
    )
    ppo_model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=lambda progress_remaining: 1e-5 + progress_remaining * (3e-4 - 1e-5),
        n_steps=n_steps,
        batch_size=1260,
        n_epochs=16,
        gamma=0.9,
        gae_lambda=0.9,
        clip_range=0.25,
        ent_coef=0.01,
        seed=run_seed,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device_used,
    )

    # Evaluation environment (single instance) for callbacks
    eval_env = DummyVecEnv([
        lambda: PortfolioEnv(val_scaled, etf_list, reward_type='log_wealth',
                             risk_coefficient=1.0, rebalance_period=rebalance,
                             lookback_period=lookback)
    ])
    eval_env.seed(run_seed)

    # Create output directories
    # Use a distinct log directory for each run to avoid overwriting files
    # when executing multiple experiments with different seeds.  If a seed
    # is provided, append it to the directory name; otherwise use the
    # default 'logs'.
    if seed is not None:
        logs_dir = Path(f'logs_seed{run_seed}')
    else:
        logs_dir = Path('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Callback to save best model and record evaluation metrics
    training_history_path = logs_dir / 'training_history.csv'
    # Ensure history file is initialised
    if training_history_path.exists():
        training_history_path.unlink()
    with open(training_history_path, 'w') as f:
        f.write('timesteps,mean_reward,timestamp\n')

    class HistoryCallback(BaseCallback):
        """Callback for logging evaluation history to CSV."""
        def __init__(self, eval_callback: EvalCallback, history_path: Path, verbose: int = 0):
            super().__init__(verbose)
            self.eval_callback = eval_callback
            self.history_path = history_path

        def _on_step(self) -> bool:
            # Evaluation happens in the wrapped EvalCallback; here we record its results
            if self.eval_callback.n_calls != 0 and self.eval_callback.n_calls % eval_freq == 0:
                # The EvalCallback stores results in last_mean_reward
                mean_reward = self.eval_callback.last_mean_reward
                with open(self.history_path, 'a') as f:
                    f.write(f"{self.num_timesteps},{mean_reward},{int(time.time())}\n")
            return True

    # Set up EvalCallback for periodic evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(logs_dir),
        log_path=str(logs_dir),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    # Checkpoint callback to save intermediate models
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=str(logs_dir),
        name_prefix='checkpoint'
    )

    # History logging callback
    history_callback = HistoryCallback(eval_callback, training_history_path)

    # Combine callbacks
    callback = CallbackList([eval_callback, checkpoint_callback, history_callback])

    # Start training
    ppo_model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the final model.  Append the seed to the filename when provided
    # to prevent collisions when running multiple experiments in the same
    # directory.  If no seed is supplied, retain the default name.
    if seed is not None:
        final_model_path = f'ppo_stage2_final_model_seed{run_seed}.zip'
    else:
        final_model_path = 'ppo_stage2_final_model.zip'
    ppo_model.save(final_model_path)

    # Load the best model (if exists) for evaluation; fall back to final model
    best_model_path = logs_dir / 'best_model.zip'
    if best_model_path.exists():
        best_model = PPO.load(best_model_path, env=eval_env, device=device_used)
    else:
        best_model = ppo_model

    # Evaluate on validation set to report final reward
    val_env = PortfolioEnv(val_scaled, etf_list, reward_type='log_wealth',
                           risk_coefficient=1.0, rebalance_period=rebalance,
                           lookback_period=lookback)
    obs, _ = val_env.reset(seed=run_seed)
    done = False
    val_reward = 0.0
    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = val_env.step(action)
        val_reward += reward

    print(f"Total validation reward: {val_reward:.4f}")

    # Out‑of‑sample testing on the remaining data
    test_env = PortfolioEnv(test_scaled, etf_list, reward_type='log_wealth',
                            risk_coefficient=1.0, rebalance_period=rebalance,
                            lookback_period=lookback)
    obs, _ = test_env.reset(seed=run_seed)
    done = False
    weights_history: list[list] = []
    while not done:
        # Produce an action every step; env will apply it only on rebalance dates
        action, _ = best_model.predict(obs, deterministic=True)
        obs, _, done, _, _ = test_env.step(action)
        # Record weights at rebalance points
        if test_env.current_step % rebalance == 0 and test_env.current_step > 0:
            date = test_scaled.loc[test_env.current_step-1, 'Date']
            weights_history.append([date] + test_env.current_weights.tolist())

    # Save monthly weights to CSV
    weights_df = pd.DataFrame(weights_history,
                              columns=['Date'] + etf_list)
    weights_csv_path = 'ppo_stage2_weights.csv'
    weights_df.to_csv(weights_csv_path, index=False)

    # Compute drifted daily returns and compare to equal weights
    def compute_returns(weights: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        common = [c for c in weights.columns if f'Price_{c}' in price_df.columns]
        price_columns = [f'Price_{c}' for c in common if f'Price_{c}' in price_df.columns]

        # Safety check
        if len(price_columns) != len(common):
            missing_cols = set(f'Price_{c}' for c in common) - set(price_columns)
            raise ValueError(f"Missing price columns for ETFs: {missing_cols}")

        price_df = price_df[price_columns]
        price_df.columns = common  # remove the Price_ prefix for subsequent calculations
        daily_returns = price_df.pct_change().dropna()
        weights = weights.set_index('Date')
        start = weights.index.min()
        end = weights.index.max() + timedelta(days=rebalance)
        daily_returns = daily_returns.loc[start:end]

        eq_weight = np.array([1/len(common)]*len(common))

        drifted = pd.DataFrame(np.nan, index=daily_returns.index, columns=common, dtype=float)
        eq_drift = pd.DataFrame(np.nan, index=daily_returns.index, columns=common, dtype=float)
        cur_w = weights.iloc[0].values
        cur_eq = eq_weight

        returns_df = pd.DataFrame(index=daily_returns.index,
                                  columns=['RL', 'Equal'])

        for d in daily_returns.index:
            rets = daily_returns.loc[d].values  # <-- Initialize here at the beginning

            if d in weights.index:
                cur_w = weights.loc[d].values
                cur_eq = eq_weight.copy()
            else:
                cur_w = cur_w * (1 + rets)
                cur_w /= cur_w.sum()
                cur_eq = cur_eq * (1 + rets)
                cur_eq /= cur_eq.sum()

            drifted.loc[d] = cur_w
            eq_drift.loc[d] = cur_eq

            shifted_rl = drifted.shift(1).loc[d].values
            shifted_eq = eq_drift.shift(1).loc[d].values

            if np.isnan(shifted_rl).any():
                returns_df.loc[d, 'RL'] = np.dot(cur_w, rets)
                returns_df.loc[d, 'Equal'] = np.dot(cur_eq, rets)
            else:
                returns_df.loc[d, 'RL'] = np.dot(shifted_rl, rets)
                returns_df.loc[d, 'Equal'] = np.dot(shifted_eq, rets)

        return returns_df.dropna()

    # Build a price DataFrame for return computation.  If `prices_file` is
    # provided, read it; otherwise construct from the full feature set.
    if prices_file is not None:
        prices_df = pd.read_csv(prices_file)
        prices_df['Date'] = pd.to_datetime(prices_df['Date'], utc=True).dt.tz_localize(None)
        prices_df.rename(columns={c: f'Price_{c}' for c in prices_df.columns if c != 'Date'}, inplace=True)
        price_df_for_returns = prices_df.set_index('Date')
    else:
        # Use the price columns from the full dataset (data_with_features)
        price_cols = [f'Price_{t}' for t in etf_list]
        price_df_for_returns = data_with_features[['Date'] + price_cols].set_index('Date')
    test_returns = compute_returns(weights_df, price_df_for_returns)
    cum_rl = (1 + test_returns['RL']).prod() - 1
    cum_equal = (1 + test_returns['Equal']).prod() - 1
    print(f"Out‑of‑sample cumulative return (RL):    {cum_rl:.4%}")
    print(f"Out‑of‑sample cumulative return (Equal): {cum_equal:.4%}")

    # Compute performance metrics
    def performance_metrics(returns: pd.Series, freq: int = 252) -> tuple[float, float, float, float]:
        ann_return = (1 + returns).prod()**(freq/len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(freq)
        sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
        cum_pnl = (1+returns).cumprod()
        max_dd = (cum_pnl / cum_pnl.cummax() - 1).min()
        return ann_return, ann_vol, sharpe, max_dd

    rl_ann, rl_vol, rl_sharpe, rl_dd = performance_metrics(test_returns['RL'])
    eq_ann, eq_vol, eq_sharpe, eq_dd = performance_metrics(test_returns['Equal'])

    print(f"RL annualised return:    {rl_ann:.4%}, Sharpe: {rl_sharpe:.3f}, Max Drawdown: {rl_dd:.4%}")
    print(f"Equal annualised return: {eq_ann:.4%}, Sharpe: {eq_sharpe:.3f}, Max Drawdown: {eq_dd:.4%}")

    return {
        'final_model_path': final_model_path,
        'best_model_path': str(best_model_path) if best_model_path.exists() else final_model_path,
        'weights_csv': weights_csv_path,
        'training_history': str(training_history_path),
        'cumulative_return_rl': cum_rl,
        'cumulative_return_equal': cum_equal,
        'rl_metrics': (rl_ann, rl_vol, rl_sharpe, rl_dd),
        'equal_metrics': (eq_ann, eq_vol, eq_sharpe, eq_dd)
    }


if __name__ == '__main__':
    # Define file paths
    STAGE2_FILE = 'stage2_rl_observations_optimized_10ETFs.csv'
    PRICES_FILE = 'stock_prices_10ETFs.csv'
    # Load and prepare data
    data_with_features = load_and_prepare_data(STAGE2_FILE, PRICES_FILE)
    # Determine list of ETFs from the price columns
    etf_columns = [c.replace('Price_', '') for c in data_with_features.columns
                   if c.startswith('Price_')]
    # Train and evaluate
    results = train_and_evaluate(
        data_with_features,
        etf_columns,
        lookback=60,
        rebalance=21,
        train_years=10,
        val_years=2,
        total_timesteps=int(6e6),
        eval_freq=int(2e5),
        device='auto',
        prices_file=PRICES_FILE
    )
    # Save results summary to JSON for future reference
    with open('training_results_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)