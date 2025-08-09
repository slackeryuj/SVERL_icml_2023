"""
Run the extended Stage 2 PPO training multiple times with different seeds.

This script demonstrates how to perform several independent training
experiments using varying random seeds.  It leverages the modified
`train_and_evaluate` function from `ppo_stage2_training_extended.py`, which
accepts a `seed` argument to override the global seed.  After each run
completes, the results and intermediate artefacts are collected into a
summary list and written to a JSON file for later analysis.

The purpose of running multiple seeds is to assess the sensitivity of the
reinforcement learning agent to initialisation and stochastic decisions.
By inspecting the output metrics (e.g. cumulative return, Sharpe ratio),
one can identify the seed that yields the most favourable out‑of‑sample
performance.  Each run writes its models, logs and history into a
dedicated directory (`logs_seed<seed>`), so no files are overwritten.

To keep demonstration runtime reasonable, this script defaults to a
reduced number of training timesteps (`1e4`) and evaluation frequency
(`2000`).  Adjust these values as needed for more thorough experiments.

Usage
-----
Execute this script directly (e.g., `python multi_seed_runner.py`).  It will
load the CSV files `stage2_rl_observations_optimized_10ETFs.csv` and
`stock_prices_10ETFs.csv` from the current working directory, run
training for each seed in the range 1–10, and produce a JSON file
`multi_seed_results.json` summarising the key statistics per seed.
"""

import json
from pathlib import Path
from datetime import timedelta

import pandas as pd

from ppo_stage2_training_extended_update import (
    load_and_prepare_data,
    train_and_evaluate,
    scale_features,
    PortfolioEnv,
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def main():
    # Define file paths
    stage2_file = 'stage2_rl_observations_optimized_10ETFs.csv'
    prices_file = 'stock_prices_10ETFs.csv'

    # Load and prepare data once
    data_with_features = load_and_prepare_data(stage2_file, prices_file)
    etf_columns = [c.replace('Price_', '') for c in data_with_features.columns if c.startswith('Price_')]

    # Define seeds to evaluate
    seeds = list(range(1, 11))  # seeds 1 through 10

    results_summary = []

    def evaluate_pretrained_model(
        full_data: pd.DataFrame,
        etfs: list[str],
        model_path: str,
        lookback: int,
        rebalance: int,
        train_years: int,
        val_years: int,
        seed: int,
        prices_file: str | None = None,
    ) -> dict:
        """Evaluate a pretrained PPO model on the validation and test sets.

        This function mirrors the evaluation logic in `train_and_evaluate` but
        skips the training phase entirely.  It splits the data into
        train/validation/test partitions, standardises features, loads the
        supplied model, and computes performance metrics on the out‑of‑sample
        period.

        Parameters
        ----------
        full_data : DataFrame
            The full dataset with engineered features.
        etfs : list of str
            List of ETF tickers to include in the environment.
        model_path : str
            Path to the pretrained PPO model to load.
        lookback : int
            Length of the observation window in trading days.
        rebalance : int
            Rebalance frequency in trading days.
        train_years : int
            Number of years used for training in the original run (only used to
            split the data for scaling purposes).
        val_years : int
            Number of years used for validation in the original run (only used
            to split the data for scaling purposes).
        seed : int
            Random seed used to initialise evaluation environments.
        prices_file : str or None
            Optional path to the raw prices CSV used for computing returns.
        """
        # Determine split lengths
        train_days = train_years * 252
        val_days = val_years * 252
        start_idx = 0
        train_data = full_data.iloc[start_idx:start_idx+train_days].reset_index(drop=True)
        val_data = full_data.iloc[start_idx+train_days:
                                  start_idx+train_days+val_days].reset_index(drop=True)
        test_data = full_data.iloc[start_idx+train_days+val_days:].reset_index(drop=True)

        # Standardise features
        train_scaled, val_scaled, test_scaled, feature_cols, _ = scale_features(train_data, val_data, test_data)

        # Create evaluation environments
        eval_env = DummyVecEnv([
            lambda: PortfolioEnv(val_scaled, etfs, reward_type='log_wealth',
                                 risk_coefficient=1.0, rebalance_period=rebalance,
                                 lookback_period=lookback)
        ])
        eval_env.seed(seed)

        # Load the pretrained model
        model = PPO.load(model_path, env=eval_env, device='auto')

        # Evaluate on validation set
        val_env = PortfolioEnv(val_scaled, etfs, reward_type='log_wealth',
                               risk_coefficient=1.0, rebalance_period=rebalance,
                               lookback_period=lookback)
        obs, _ = val_env.reset(seed=seed)
        done = False
        val_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = val_env.step(action)
            val_reward += reward

        # Evaluate on the out‑of‑sample test set
        test_env = PortfolioEnv(test_scaled, etfs, reward_type='log_wealth',
                                risk_coefficient=1.0, rebalance_period=rebalance,
                                lookback_period=lookback)
        obs, _ = test_env.reset(seed=seed)
        done = False
        weights_history: list[list] = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = test_env.step(action)
            if test_env.current_step % rebalance == 0 and test_env.current_step > 0:
                date = test_scaled.loc[test_env.current_step-1, 'Date']
                weights_history.append([date] + test_env.current_weights.tolist())

        # Save weights to CSV
        weights_df = pd.DataFrame(weights_history,
                                  columns=['Date'] + etfs)
        weights_csv_path = f'ppo_stage2_weights_seed{seed}.csv'
        weights_df.to_csv(weights_csv_path, index=False)

        # Compute returns using the helper from the training script
        # Recreate the price DataFrame
        if prices_file is not None:
            prices_df = pd.read_csv(prices_file)
            prices_df['Date'] = pd.to_datetime(prices_df['Date'], utc=True).dt.tz_localize(None)
            prices_df.rename(columns={c: f'Price_{c}' for c in prices_df.columns if c != 'Date'}, inplace=True)
            price_df_for_returns = prices_df.set_index('Date')
        else:
            price_cols = [f'Price_{t}' for t in etfs]
            price_df_for_returns = full_data[['Date'] + price_cols].set_index('Date')

        # Compute drifted returns and metrics (using a local implementation of compute_returns)
        def compute_returns(weights: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
            common = [c for c in weights.columns if f'Price_{c}' in price_df.columns]
            price_columns = [f'Price_{c}' for c in common if f'Price_{c}' in price_df.columns]
            if len(price_columns) != len(common):
                missing_cols = set(f'Price_{c}' for c in common) - set(price_columns)
                raise ValueError(f"Missing price columns for ETFs: {missing_cols}")
            price_df = price_df[price_columns]
            price_df.columns = common
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
                rets = daily_returns.loc[d].values
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

        test_returns = compute_returns(weights_df, price_df_for_returns)
        cum_rl = (1 + test_returns['RL']).prod() - 1
        cum_equal = (1 + test_returns['Equal']).prod() - 1
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
        # Compose result dictionary
        return {
            'final_model_path': model_path,
            'best_model_path': model_path,
            'weights_csv': weights_csv_path,
            'training_history': None,
            'cumulative_return_rl': cum_rl,
            'cumulative_return_equal': cum_equal,
            'rl_metrics': (rl_ann, rl_vol, rl_sharpe, rl_dd),
            'equal_metrics': (eq_ann, eq_vol, eq_sharpe, eq_dd)
        }

    for seed in seeds:
        print(f"\n===== Starting run with seed {seed} =====")
        # Determine the expected filename for a previously trained model.
        model_filename = f'ppo_stage2_final_model_seed{seed}.zip'
        model_path = Path(model_filename)

        if model_path.exists():
            print(f"Found existing model for seed {seed}: {model_filename}. Skipping training and loading it.")
            # Evaluate using the pretrained model by supplying the path to
            # train_and_evaluate.  The function will bypass training and
            # perform only evaluation when a valid pretrained_model_path is
            # provided.
            results = train_and_evaluate(
                data_with_features,
                etf_columns,
                lookback=21,
                rebalance=10,
                train_years=8,
                val_years=2,
                total_timesteps=int(5e6),
                eval_freq=int(2e5),
                device='auto',
                prices_file=prices_file,
                seed=seed,
                pretrained_model_path=model_filename,
            )
        else:
            # No existing model, perform training from scratch
            results = train_and_evaluate(
                data_with_features,
                etf_columns,
                lookback=21,
                rebalance=10,
                train_years=8,
                val_years=2,
                total_timesteps=int(5e6),
                eval_freq=int(2e5),
                device='auto',
                prices_file=prices_file,
                seed=seed,
            )
        # Attach the seed to the results for later identification
        results['seed'] = seed
        results_summary.append(results)

    # Write the aggregated results to JSON
    summary_path = Path('multi_seed_results.json')
    with summary_path.open('w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"\nAll runs complete.  Summary written to {summary_path}.")


if __name__ == '__main__':
    main()