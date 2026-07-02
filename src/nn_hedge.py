"""Simple neural network that predicts option delta hedge ratios."""

import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.features import FEATURE_COLUMNS, add_market_features, add_realized_delta_target
from src.utils import set_random_seeds


def prepare_nn_dataset(options_df, stock_df, r):
    """Create features and realized dC/dS targets, with a proxy fallback."""
    featured = add_market_features(options_df, stock_df, r)
    targeted = add_realized_delta_target(featured)
    if targeted.empty:
        print(
            "WARNING: Realized next-period option changes are unavailable. "
            "Training the NN on Black-Scholes delta as a placeholder target."
        )
        targeted = featured.copy()
        targeted["realized_delta"] = targeted["option_delta_bs"]
        targeted["target_is_proxy"] = True
    else:
        targeted["target_is_proxy"] = False
    return targeted


class HedgeNet(nn.Module):
    """Beginner-readable feedforward delta-regression network."""

    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_nn_hedge_model(
    dataset,
    epochs=300,
    learning_rate=1e-3,
    patience=35,
    random_seed=42,
    model_path="results/nn_hedge_model.pt",
):
    """Train using the earliest 70% of dates and validate on the next 15%."""
    set_random_seeds(random_seed)
    data = dataset.sort_values("date").reset_index(drop=True)
    train_end = max(2, int(len(data) * 0.70))
    validation_end = max(train_end + 1, int(len(data) * 0.85))
    if validation_end >= len(data):
        validation_end = len(data) - 1
    train = data.iloc[:train_end]
    validation = data.iloc[train_end:validation_end]
    if validation.empty:
        validation = train.tail(max(1, len(train) // 5))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLUMNS])
    X_validation = scaler.transform(validation[FEATURE_COLUMNS])
    y_train = train["realized_delta"].to_numpy()
    y_validation = validation["realized_delta"].to_numpy()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_validation = torch.tensor(X_validation, dtype=torch.float32)
    y_validation = torch.tensor(y_validation, dtype=torch.float32).reshape(-1, 1)

    model = HedgeNet(len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    waits = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(model(X_validation), y_validation).item()
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            waits = 0
        else:
            waits += 1
        if waits >= patience:
            print(f"NN early stopping after {epoch + 1} epochs.")
            break

    model.load_state_dict(best_state)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        model_path,
    )
    test_start_date = data.iloc[validation_end]["date"]
    return model, scaler, test_start_date


def predict_nn_delta(model, scaler, features):
    """Predict clipped option deltas from unscaled features."""
    model.eval()
    X = torch.tensor(scaler.transform(features[FEATURE_COLUMNS]), dtype=torch.float32)
    with torch.no_grad():
        prediction = model(X).numpy().ravel()
    return np.clip(prediction, -2.0, 2.0)
