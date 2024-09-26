# code_snippets/scikit_learn/model_training.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from utilities.common_functions import load_config
import joblib

def train_random_forest(X_train, y_train, config=None):
    """
    Train a Random Forest classifier.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        config (dict): Configuration dictionary.

    Returns:
        model (sklearn.ensemble.RandomForestClassifier): Trained model.
    """
    if config is None:
        config = load_config()
    
    n_estimators = config['model'].get('n_estimators', 100)
    max_depth = config['model'].get('max_depth', None)
    random_state = config['model'].get('random_state', 42)
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model_sklearn(model, X_test, y_test):
    """
    Evaluate the scikit-learn model on test data.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f'Accuracy: {acc:.4f}')
    print('Classification Report:')
    print(report)

def save_model_sklearn(model, filepath):
    """
    Save the scikit-learn model to a file.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained model.
        filepath (str): Path to save the model.

    Returns:
        None
    """
    joblib.dump(model, filepath)
    print(f'Model saved to {filepath}')

def load_model_sklearn(filepath):
    """
    Load a scikit-learn model from a file.

    Args:
        filepath (str): Path to the saved model.

    Returns:
        model (sklearn.ensemble.RandomForestClassifier): Loaded model.
    """
    model = joblib.load(filepath)
    print(f'Model loaded from {filepath}')
    return model
