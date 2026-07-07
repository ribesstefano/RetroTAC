"""
torch_common.py
===============
Shared PyTorch training plumbing used by the MLP and GNN models.
Depends on torch (kept separate from mol_utils so the XGBoost path stays torch-free).
"""
import random
 
import numpy as np
import torch
from torch.utils.data import Dataset
 
 
def get_device(preference: str = "auto") -> torch.device:
    """Return the best available device.
 
    preference: 'auto' probes hardware (cuda -> mps -> cpu); or force
    'cuda' / 'mps' / 'cpu' explicitly (e.g. from config).
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
 
 
def seed_everything(seed: int) -> None:
    """Seed all RNGs the training loop touches for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # no-op if no CUDA device
 
 
class EarlyStopper:
    """Watch validation loss; signal when to stop and remember the best weights."""
 
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None
 
    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Call once per epoch. Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience
 
    def restore(self, model: torch.nn.Module) -> None:
        """Load the best-epoch weights back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
 
 
class TabularDataset(Dataset):
    """Wrap a preprocessed feature matrix + targets as a torch Dataset."""
 
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
 
    def __len__(self):
        return self.X.shape[0]
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]