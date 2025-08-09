from SVERL_icml_2023.portfolio_DRL.data_function import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from SVERL_icml_2023.portfolio_DRL.create_model import *

def main():
    # Example usage
    input_dir = "E:\XAI\RL_with_SHAP\pythonProject1\SVERL_icml_2023\portfolio_DRL"  # Use the specified directory
    ticker_file = os.path.join(input_dir, "tickers.txt")  # Text file containing tickers, one per line
    factor_file = os.path.join(input_dir, "FF_factors.csv")  # CSV file containing factor returns
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    output_csv = os.path.join(input_dir, "stock_prices.csv")

    try:
        tickers = load_tickers_from_file(ticker_file)
        price_data = download_stock_prices(tickers, start_date, end_date, output_csv)
        daily_returns = calculate_daily_returns(price_data, return_type='arithmetic')
        daily_returns.to_csv(os.path.join(input_dir, "daily_returns.csv"))
        factor_returns = read_factor_returns(factor_file, daily_returns)
        factor_returns.to_csv(os.path.join(input_dir, "aligned_factors.csv"))
        print("Factor returns alignment completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

    factors = pd.read_csv(f"{input_dir}/aligned_factors.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(f"{input_dir}/daily_returns.csv", index_col=0, parse_dates=True)

    sequence_length = 30
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = StockDataset(factors, returns, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    d_model = factors.shape[1] + returns.shape[1]
    num_heads = 4
    num_stocks = returns.shape[1]

    model = StockPredictAgent(d_model=d_model, num_heads=num_heads, sequence_length=sequence_length, num_stocks=num_stocks).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_model(model, dataloader, optimizer, criterion, num_epochs, device)

    torch.save(model.state_dict(), f"{input_dir}/stock_predict_agent.pth")
    print("Model training completed and saved.")

if __name__ == "__main__":
    main()
