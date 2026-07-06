"""PyTorch multilayer perceptron wrapped as a sklearn classifier."""

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _BinaryMLP(nn.Module):
    """Compact feed-forward network used for the MLP benchmark."""

    def __init__(self, input_features: int, hidden_layers: Sequence[int], dropout: Sequence[float]):
        super().__init__()
        layers: list[nn.Module] = []
        previous_features = input_features
        for hidden_units, dropout_probability in zip(hidden_layers, dropout, strict=True):
            layers.extend(
                [
                    nn.Linear(previous_features, hidden_units),
                    nn.BatchNorm1d(hidden_units),
                    nn.ReLU(),
                    nn.Dropout(dropout_probability),
                ]
            )
            previous_features = hidden_units
        layers.append(nn.Linear(previous_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one logit per observation."""
        return self.network(features).squeeze(-1)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Binary PyTorch MLP with AdamW, plateau scheduling, and early stopping."""

    uses_validation_data = True

    def __init__(
        self,
        hidden_layers: Sequence[int] = (128, 64),
        dropout: Sequence[float] = (0.25, 0.20),
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 512,
        max_epochs: int = 100,
        early_stopping_patience: int = 18,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        min_delta: float = 0.0,
        device: str = "cuda",
        random_state: int = 42,
    ):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_delta = min_delta
        self.device = device
        self.random_state = random_state

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            message = "PyTorch CUDA was requested, but CUDA is not available"
            raise ValueError(message)
        return torch.device(self.device)

    def _validate_architecture(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        hidden_layers = tuple(int(units) for units in self.hidden_layers)
        dropout = tuple(float(probability) for probability in self.dropout)
        if not hidden_layers:
            message = "hidden_layers must contain at least one layer"
            raise ValueError(message)
        if len(hidden_layers) != len(dropout):
            message = "dropout must contain one probability per hidden layer"
            raise ValueError(message)
        if any(units < 1 for units in hidden_layers):
            message = "hidden layer sizes must be positive"
            raise ValueError(message)
        if any(not 0 <= probability < 1 for probability in dropout):
            message = "dropout probabilities must be in [0, 1)"
            raise ValueError(message)
        return hidden_layers, dropout

    def _set_reproducible_seeds(self) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _prepare_arrays(self, features: Any, target: Any | None = None) -> tuple[np.ndarray, ...]:
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.ndim != 2:
            message = "features must be a two-dimensional array"
            raise ValueError(message)
        if target is None:
            return (feature_array,)

        target_array = np.asarray(target, dtype=np.float32)
        if not np.isin(target_array, [0.0, 1.0]).all():
            message = "TorchMLPClassifier only supports binary targets encoded as 0 and 1"
            raise ValueError(message)
        return feature_array, target_array

    def _validation_average_precision(
        self,
        features: np.ndarray,
        target: np.ndarray,
        device: torch.device,
    ) -> float:
        self.model_.eval()
        with torch.no_grad():
            feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            probabilities = torch.sigmoid(self.model_(feature_tensor)).detach().cpu().numpy()
        return average_precision_score(target, probabilities)

    def fit(
        self,
        features: Any,
        target: Any,
        validation_data: tuple[Any, Any] | None = None,
    ) -> "TorchMLPClassifier":
        """Fit the network, optionally using validation average precision for early stopping."""
        hidden_layers, dropout = self._validate_architecture()
        self._set_reproducible_seeds()
        device = self._resolve_device()

        feature_array, target_array = self._prepare_arrays(features, target)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = feature_array.shape[1]

        positive_count = float(target_array.sum())
        negative_count = float(len(target_array) - positive_count)
        if positive_count == 0 or negative_count == 0:
            message = "TorchMLPClassifier requires both binary classes in the training data"
            raise ValueError(message)
        positive_weight = torch.tensor(negative_count / positive_count, dtype=torch.float32)

        self.model_ = _BinaryMLP(self.n_features_in_, hidden_layers, dropout).to(device)
        loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight.to(device))
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        dataset = TensorDataset(
            torch.as_tensor(feature_array, dtype=torch.float32),
            torch.as_tensor(target_array, dtype=torch.float32),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        validation_arrays = None
        if validation_data is not None:
            validation_arrays = self._prepare_arrays(*validation_data)

        best_score = -np.inf
        best_state = deepcopy(self.model_.state_dict())
        epochs_without_improvement = 0
        self.training_history_ = []

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            epoch_losses = []
            for batch_features, batch_target in loader:
                batch_features = batch_features.to(device)
                batch_target = batch_target.to(device)
                optimizer.zero_grad()
                logits = self.model_(batch_features)
                loss = loss_function(logits, batch_target)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            if validation_arrays is None:
                score = -float(np.mean(epoch_losses))
            else:
                validation_features, validation_target = validation_arrays
                score = self._validation_average_precision(
                    validation_features,
                    validation_target,
                    device,
                )
            scheduler.step(score)
            self.training_history_.append({"epoch": epoch, "score": score})

            if score > best_score + self.min_delta:
                best_score = score
                best_state = deepcopy(self.model_.state_dict())
                self.best_epoch_ = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stopping_patience:
                    break

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_score_ = float(best_score)
        self.n_epochs_ = len(self.training_history_)
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        """Return class probabilities as ``[P(class=0), P(class=1)]``."""
        self.model_.eval()
        device = next(self.model_.parameters()).device
        (feature_array,) = self._prepare_arrays(features)
        with torch.no_grad():
            feature_tensor = torch.as_tensor(feature_array, dtype=torch.float32, device=device)
            positive_probabilities = torch.sigmoid(self.model_(feature_tensor)).cpu().numpy()
        return np.column_stack((1.0 - positive_probabilities, positive_probabilities))

    def predict(self, features: Any) -> np.ndarray:
        """Predict binary labels at the default 0.5 threshold."""
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)
