# code_snippets/scikit_learn/data_preprocessing.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    y = np.random.randint(0, output_dim, size=(num_samples, 1))  # for classification
    
    return X, y

def preprocess_data(X, y):
    """
    Preprocess the data: scaling and encoding.

    Args:
        X (numpy.ndarray): Features.
        y (numpy.ndarray): Labels.

    Returns:
        X_scaled (numpy.ndarray): Scaled features.
        y_encoded (numpy.ndarray): One-hot encoded labels.
        scaler (StandardScaler): Fitted scaler.
        encoder (OneHotEncoder): Fitted encoder.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y)
    
    return X_scaled, y_encoded, scaler, encoder

def split_data(X, y, config=None):
    """
    Split the data into training, validation, and test sets.

    Args:
        X (numpy.ndarray): Features.
        y (numpy.ndarray): Labels.
        config (dict): Configuration dictionary.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
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
    
    return X_train, X_val, X_test, y_train, y_val, y_test
