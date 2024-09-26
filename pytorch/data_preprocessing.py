# code_snippets/pytorch/data_preprocessing.py

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from utilities.common_functions import load_config

def load_data():
    """
    Load your dataset here.
    Replace this function with actual data loading code.
    
    Returns:
        X (numpy.ndarray): Features.
        y (numpy.ndarray): Labels.
    """
    # Example: generate random data
    config = load_config()
    num_samples = 150
    input_dim = config['model']['input_dim']
    output_dim = config['model']['output_dim']
    
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, output_dim, size=(num_samples,))  # for classification
    
    return X, y

def preprocess_data(X, y):
    """
    Preprocess the data: scaling.

    Args:
        X (numpy.ndarray): Features.
        y (numpy.ndarray): Labels.

    Returns:
        X_scaled (numpy.ndarray): Scaled features.
        y (numpy.ndarray): Labels as integers.
        scaler (StandardScaler): Fitted scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def split_data(X, y, config=None):
    """
    Split the data into training, validation, and test sets.

    Args:
        X (numpy.ndarray): Features.
        y (numpy.ndarray): Labels.
        config (dict): Configuration dictionary.

    Returns:
        train_dataset, val_dataset, test_dataset (TensorDataset)
    """
    if config is None:
        config = load_config()
    
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=42, stratify=y
    )
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val_ratio), random_state=42, stratify=y_temp
    )
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, val_dataset, test_dataset
