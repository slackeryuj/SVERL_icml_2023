import pandas as pd
import numpy as np
import torch
from SVERL_icml_2023.shapley import Shapley
from SVERL_icml_2023.utils import mask_state

class StockPredictionWrapper:
    def __init__(self, model, device):
        """
        Wrapper for the trained stock prediction model to calculate characteristic values.

        Parameters:
        model (nn.Module): Trained stock prediction model.
        device (torch.device): Device to run the model on.
        """
        self.model = model
        self.device = device

    def characteristic_values(self, states, coalition):
        """
        Compute characteristic values for given states and coalition.

        Parameters:
        states (np.array): Array of input states.
        coalition (list): Coalition of features to retain.

        Returns:
        dict: Characteristic values for the coalition.
        """
        masked_states = mask_state(states, states.shape[-1], coalition)
        factors, returns = masked_states[:, :, :len(factors)], masked_states[:, :, len(factors):]
        factors = torch.tensor(factors, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.model(factors, returns).cpu().numpy()

        return predictions

if __name__ == "__main__":
    # Paths and device
    input_dir = "/path/to/your/directory"
    factors_path = f"{input_dir}/aligned_factors.csv"
    returns_path = f"{input_dir}/daily_returns.csv"
    model_path = f"{input_dir}/stock_predict_agent.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    factors = pd.read_csv(factors_path, index_col=0, parse_dates=True)
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    states = np.concatenate([factors.values, returns.values], axis=1)

    # Load model
    d_model = factors.shape[1] + returns.shape[1]
    sequence_length = 30
    num_stocks = returns.shape[1]
    num_heads = 4

    model = StockPredictAgent(d_model=d_model, num_heads=num_heads, sequence_length=sequence_length, num_stocks=num_stocks)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Prepare Shapley calculation
    states_to_explain = states[:100]  # Example subset to explain
    shapley_calc = Shapley(states_to_explain)

    wrapper = StockPredictionWrapper(model, device)

    characteristic_values = {}
    shapley_time_series = []

    for coalition in range(1 << len(states_to_explain[0])):
        coalition_values = wrapper.characteristic_values(states_to_explain, coalition)
        characteristic_values[tuple(coalition)] = coalition_values
        shapley_time_series.append((coalition, coalition_values))

    shapley_values = shapley_calc.run(characteristic_values)

    # Save intermediate and final results
    intermediate_path = f"{input_dir}/intermediate_shapley.csv"
    shapley_results_path = f"{input_dir}/shapley_values.csv"
    shapley_time_series_path = f"{input_dir}/shapley_time_series.csv"

    pd.DataFrame.from_dict(characteristic_values, orient="index").to_csv(intermediate_path)
    pd.DataFrame(shapley_values).to_csv(shapley_results_path)
    pd.DataFrame(shapley_time_series, columns=["Coalition", "CharacteristicValues"]).to_csv(shapley_time_series_path)

    print(f"Intermediate characteristic values saved to {intermediate_path}")
    print(f"Shapley values saved to {shapley_results_path}")
    print(f"Shapley time series saved to {shapley_time_series_path}")
