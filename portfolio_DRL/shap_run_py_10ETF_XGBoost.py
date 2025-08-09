# stage 1 training and prediction
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import ta
import joblib
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import random
import os
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# ------------------------------------------------------------------------
# 0. Load and align data
# ------------------------------------------------------------------------
factors = pd.read_csv("aligned_factors.csv", index_col=0, parse_dates=True)
returns = pd.read_csv("daily_returns_10ETFs.csv", index_col=0, parse_dates=True)

# Align dates to ensure matching indices
dates = factors.index.intersection(returns.index)
factors = factors.loc[dates]
returns = returns.loc[dates]

# ------------------------------------------------------------------------
# 1. Compute technical indicators and lagged features per ETF
# ------------------------------------------------------------------------
all_tech_features = []

for etf in returns.columns:
    close = (1 + returns[etf]).cumprod()
    tech_df = pd.DataFrame(index=returns.index)

    # Selected indicators (others commented out to reduce noise)
    tech_df[f'{etf}_SMA_5'] = ta.trend.sma_indicator(close, window=5)
    tech_df[f'{etf}_EMA_12'] = ta.trend.ema_indicator(close, window=12)
    tech_df[f'{etf}_RSI_7'] = ta.momentum.rsi(close, window=7)
    tech_df[f'{etf}_MACD'] = ta.trend.macd_diff(close)
    tech_df[f'{etf}_ATR'] = ta.volatility.average_true_range(
        high=close * 1.01, low=close * 0.99, close=close, window=10
    )
    tech_df[f'{etf}_Vol_5'] = returns[etf].rolling(window=5).std()
    tech_df[f'{etf}_Mom_3'] = returns[etf].rolling(window=3).mean()

    # Lagged returns (shifted so only past information is used)
    for lag in [1, 2, 3]:
        tech_df[f'{etf}_LagRet_{lag}'] = returns[etf].shift(lag)

    all_tech_features.append(tech_df)

# Concatenate technical indicators for all ETFs
technical_features = pd.concat(all_tech_features, axis=1)

# ------------------------------------------------------------------------
# 2. Create lagged factor features
# ------------------------------------------------------------------------
for factor in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
    for lag in [1, 2, 3]:
        factors[f'{factor}_lag_{lag}'] = factors[factor].shift(lag)

# Drop rows with NA values arising from lagging
factors = factors.dropna()

# ------------------------------------------------------------------------
# 3. Combine factors, technical features, and VIX change
# ------------------------------------------------------------------------
features = pd.concat([factors, technical_features], axis=1).dropna()
vix = pd.read_csv("VIX_History.csv", index_col=0, parse_dates=True)

# Align VIX to our feature dates and compute lagged change
vix_aligned = vix['CLOSE'].reindex(features.index).ffill()
features['VIX'] = vix_aligned.pct_change(fill_method=None).shift(1)
features['VIX'] = features['VIX'].fillna(0)

# Define the target: next-day return per ETF
target_returns = returns.shift(-1).loc[features.index].dropna()
features = features.loc[target_returns.index]

# ------------------------------------------------------------------------
# 4. Define rolling window parameters
# ------------------------------------------------------------------------
train_years = 12  # years used for training
valid_years = 1  # years used for validation
test_years = 1  # years used for testing/prediction
retrain_frequency = 1  # years between retrainings
start_year = 2009
end_year = 2024

# List generic features used for SHAP importance ranking
all_generic_features = [
    'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA',
    'Mkt-RF_lag_1', 'Mkt-RF_lag_2', 'Mkt-RF_lag_3',
    'SMB_lag_1', 'SMB_lag_2', 'SMB_lag_3',
    'HML_lag_1', 'HML_lag_2', 'HML_lag_3',
    'RMW_lag_1', 'RMW_lag_2', 'RMW_lag_3',
    'CMA_lag_1', 'CMA_lag_2', 'CMA_lag_3',
    'SMA_5', 'EMA_12', 'RSI_7', 'MACD',
    'Vol_5', 'Mom_3',
    'LagRet_1', 'LagRet_2', 'LagRet_3', 'VIX'
]

# ------------------------------------------------------------------------
# 5. Compute generic feature importance via SHAP
#    (aggregated across ETFs, using only the initial training window)
# ------------------------------------------------------------------------
shap_importances = pd.DataFrame(0.0, index=all_generic_features, columns=['SHAP_Value'])

# Use a fixed period (e.g. up to year 2009) for computing importances
base_train_start = pd.Timestamp(start_year - train_years, 1, 1)
base_train_end = pd.Timestamp(start_year - valid_years - 1, 12, 31)

for etf in returns.columns:
    print(f"Computing SHAP importances for ETF: {etf}")
    # Filter columns relevant to this ETF (generic + factor features)
    etf_cols = [
        col for col in features.columns
        if (etf in col and any(k in col for k in ['SMA_5', 'EMA_12', 'RSI_7',
                                                  'MACD', 'Vol_5', 'Mom_3',
                                                  'LagRet_1', 'LagRet_2', 'LagRet_3', 'VIX']))
           or col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA',
                      'Mkt-RF_lag_1', 'Mkt-RF_lag_2', 'Mkt-RF_lag_3',
                      'SMB_lag_1', 'SMB_lag_2', 'SMB_lag_3',
                      'HML_lag_1', 'HML_lag_2', 'HML_lag_3',
                      'RMW_lag_1', 'RMW_lag_2', 'RMW_lag_3',
                      'CMA_lag_1', 'CMA_lag_2', 'CMA_lag_3']
    ]
    X_base = features.loc[base_train_start:base_train_end, etf_cols]
    y_base = target_returns[etf].loc[base_train_start:base_train_end]

    # Fit a quick model to compute SHAP
    model_base = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=GLOBAL_SEED,
        seed=GLOBAL_SEED,
        device='cuda'
    ).fit(X_base, y_base)

    explainer_base = shap.Explainer(model_base)
    shap_vals = explainer_base(X_base)

    # Aggregate SHAP values per generic feature
    for gen_feat in all_generic_features:
        cols = [c for c in X_base.columns if gen_feat in c]
        if cols:
            idx = [X_base.columns.get_loc(c) for c in cols]
            shap_importances.loc[gen_feat] += np.mean(np.abs(shap_vals.values[:, idx]))

# Average importance across ETFs and select top N
shap_importances /= len(returns.columns)
top_generic_features = (
    shap_importances.sort_values('SHAP_Value', ascending=False)
    .head(10)
    .index
    .tolist()
)

# returns = returns.iloc[:,:2]
# ------------------------------------------------------------------------
# 6. Retrain models using the selected generic features in rolling windows
# ------------------------------------------------------------------------
all_predictions = []

# for etf in returns.columns:
#     print(f"\n==== Training models for ETF: {etf} ====")
#     # Select columns containing any of the top_generic_features or factor names
#     selected_features = [
#         f for f in features.columns
#         if any(gen in f for gen in top_generic_features) or f in factors.columns
#     ]

for etf in returns.columns:
    print(f"\n==== Training models for ETF: {etf} ====")

    # Select features explicitly relevant to current ETF
    # selected_features = [
    #     f for f in features.columns
    #     if (
    #         # Include ETF-specific technical indicators explicitly
    #         (any(gen in f for gen in top_generic_features) and (etf in f))
    #         # Include ONLY explicitly selected generic FF factors or their lags
    #         or (any(gen == f for gen in top_generic_features))
    #     )
    # ]

    selected_features = []

    for feature in top_generic_features:
        # Clearly check if the feature is ETF-specific (technical indicators)
        etf_specific_feature_name = f'{etf}_{feature}'

        # Add ETF-specific feature explicitly if present in columns
        if etf_specific_feature_name in features.columns:
            selected_features.append(etf_specific_feature_name)

        # If not ETF-specific, explicitly add generic factor features directly
        elif feature in features.columns:
            selected_features.append(feature)

    # Sanity check to ensure you have valid selected features
    if not selected_features:
        raise ValueError(f"No features selected for {etf}, please verify feature names.")

    print(f"Selected features for {etf}: {selected_features}")

    year = start_year
    while year <= end_year - test_years + 1:
        print(f"\nTraining window starting {year}")
        start_time = time.time()

        # Define periods
        train_start = pd.Timestamp(year - train_years, 1, 1)
        train_end = pd.Timestamp(year - valid_years - 1, 12, 31)
        valid_start = pd.Timestamp(year - valid_years, 1, 1)
        valid_end = pd.Timestamp(year - 1, 12, 31)
        test_start = pd.Timestamp(year, 1, 1)
        test_end = pd.Timestamp(year + test_years - 1, 12, 31)

        # Extract data
        X_train = features.loc[train_start:train_end, selected_features]
        y_train = target_returns[etf].loc[train_start:train_end]
        X_valid = features.loc[valid_start:valid_end, selected_features]
        y_valid = target_returns[etf].loc[valid_start:valid_end]
        X_test = features.loc[test_start:test_end, selected_features]
        y_test = target_returns[etf].loc[test_start:test_end]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        # # Base model with early stopping
        # base_model = xgb.XGBRegressor(
        #     objective='reg:squarederror',
        #     tree_method='hist',
        #     device='cuda',
        #     random_state=42,
        #     n_jobs=4,
        #     # eval_metric='rmse', # The metric to monitor for early stopping
        #     # early_stopping_rounds=50
        # )
        #
        # param_grid = {
        #     'n_estimators': [200, 400],
        #     'max_depth': [3, 4, 5],
        #     'learning_rate': [0.03, 0.05],
        #     'subsample': [0.7, 0.8],
        #     'colsample_bytree': [0.7, 0.8]
        # }
        #
        # tscv = TimeSeriesSplit(n_splits=3)
        #
        # grid_search = GridSearchCV(
        #     base_model,
        #     param_grid,
        #     cv=tscv,
        #     scoring='neg_mean_squared_error',
        #     verbose=0,
        #     n_jobs=4
        # )
        #
        # # Fit with early stopping on the explicit validation set
        # # fit_params = {
        # #     "eval_set": [(X_valid, y_valid)],
        # #     "verbose": False
        # # }
        # #
        # # # grid_search.fit(X_train, y_train, **fit_params)
        # # grid_search.fit(X_train_scaled, y_train)
        #
        # fit_params = {
        #     'eval_set': [(X_valid_scaled, y_valid)],
        #     'eval_metric': 'rmse',
        #     'early_stopping_rounds': 50,
        #     'verbose': False
        # }
        #
        # grid_search.fit(X_train_scaled, y_train, **fit_params)
        #
        # best_model = grid_search.best_estimator_
        #
        # # Predict on the test period
        # preds = best_model.predict(X_test_scaled)

        # Define your parameter grid explicitly
        param_grid = {
            'n_estimators': [200, 400],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.03, 0.05],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8]
        }

        tscv = TimeSeriesSplit(n_splits=3)

        best_score = float('inf')
        best_params = None
        best_model = None

        # Explicit loop for parameter search and cross-validation
        for params in ParameterGrid(param_grid):
            cv_rmse = []

            for train_idx, val_idx in tscv.split(X_train_scaled):
                X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # DMatrix explicitly required by XGBoost native API
                dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
                dval = xgb.DMatrix(X_fold_val, label=y_fold_val)

                # Setup watchlist explicitly for early stopping
                watchlist = [(dtrain, 'train'), (dval, 'validation')]

                xgb_params = {
                    'objective': 'reg:squarederror',
                    'tree_method': 'hist',
                    'device': 'cuda',
                    'eval_metric': 'rmse',
                    'seed': 42,
                    'max_depth': params['max_depth'],
                    'learning_rate': params['learning_rate'],
                    'subsample': params['subsample'],
                    'colsample_bytree': params['colsample_bytree']
                }

                # Explicitly train with early stopping
                model = xgb.train(
                    xgb_params,
                    dtrain,
                    num_boost_round=params['n_estimators'],
                    evals=watchlist,
                    early_stopping_rounds=50,
                    verbose_eval=False
                )

                preds = model.predict(dval)
                rmse = np.sqrt(mean_squared_error(y_fold_val, preds))
                cv_rmse.append(rmse)

            avg_rmse = np.mean(cv_rmse)
            # print(f"Params: {params}, CV Avg RMSE: {avg_rmse:.6f}")

            if avg_rmse < best_score:
                best_score = avg_rmse
                best_params = params
                best_model = model

        # Train final model explicitly with best parameters on full training data
        dtrain_full = xgb.DMatrix(X_train_scaled, label=y_train)
        dvalid_full = xgb.DMatrix(X_valid_scaled, label=y_valid)

        watchlist_full = [(dtrain_full, 'train'), (dvalid_full, 'validation')]

        final_xgb_params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': 'rmse',
            'seed': 42,
            'max_depth': best_params['max_depth'],
            'learning_rate': best_params['learning_rate'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree']
        }

        best_model = xgb.train(
            final_xgb_params,
            dtrain_full,
            num_boost_round=best_params['n_estimators'],
            evals=watchlist_full,
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Predict explicitly on test data
        dtest = xgb.DMatrix(X_test_scaled)
        preds = best_model.predict(dtest)

        # Metrics clearly
        test_rmse = np.sqrt(mean_squared_error(y_test, preds))
        # print(f"Test RMSE: {test_rmse:.6f}")

        # Compute evaluation metrics
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        dir_acc = np.mean((np.sign(y_test) == np.sign(preds)).astype(int))

        print(f"MSE: {mse:.6f}  RMSE: {rmse:.6f}  MAE: {mae:.6f}  "
              f"R²: {r2:.6f}  DirAcc: {dir_acc:.2%}")

        # Save the model for reproducibility
        joblib.dump(best_model, f"best_model_{etf}_{year}.joblib")

        # Save predictions
        preds_df = pd.DataFrame({
            'Date': X_test.index,
            'ETF': etf,
            'Year': year,
            'Actual_Return': y_test,
            'Predicted_Return': preds
        }).reset_index(drop=True)

        # Compute SHAP values on the test set
        explainer_test = shap.Explainer(best_model, feature_names=X_test.columns)
        shap_vals_test = explainer_test(X_test_scaled)

        clean_shap_cols = [
            f'SHAP_{col.replace(f"{etf}_", "")}' if col.startswith(f'{etf}_') else f'SHAP_{col}'
            for col in X_test.columns
        ]

        shap_df = pd.DataFrame(
            shap_vals_test.values,
            columns=clean_shap_cols,
            # columns=[f'SHAP_{col}' for col in X_test.columns],
            index=X_test.index
        ).reset_index().rename(columns={'index': 'Date'})

        # Merge SHAP values with predictions
        preds_df = preds_df.merge(shap_df, on='Date', how='left')

        all_predictions.append(preds_df)

        # Advance the window
        year += retrain_frequency
        print(f"Window processed in {time.time() - start_time:.2f} seconds")

# Concatenate and save all predictions and SHAP values
final_predictions_df = pd.concat(all_predictions, ignore_index=True)
final_predictions_df.to_csv("stage1_predictions_with_shap_10ETFs.csv", index=False)

print("Stage 1 completed and data saved for Stage 2.")
