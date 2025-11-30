flowchart LR

    X[Feature vector X i t with tech features and factor lags]
    M[XGBoost model ensemble]
    O[Outputs: predicted return score and shap groups]

    X --> M --> O
