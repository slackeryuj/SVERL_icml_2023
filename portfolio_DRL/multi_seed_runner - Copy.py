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

import pandas as pd

from ppo_stage2_training_extended_update import (
    load_and_prepare_data,
    train_and_evaluate,
)


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

    for seed in seeds:
        print(f"\n===== Starting run with seed {seed} =====")
        # Run training and evaluation with a smaller number of timesteps
        results = train_and_evaluate(
            data_with_features,
            etf_columns,
            lookback=21,
            rebalance=10,
            train_years=8,  # shorter training window for demonstration
            val_years=2,    # shorter validation window for demonstration
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