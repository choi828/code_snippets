# code_snippets/keras/evaluation.py

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
from utilities.common_functions import load_config

def evaluate_keras_model(model_path, X_test, y_test, config=None):
    """
    Evaluate the Keras model on test data.

    Args:
        model_path (str): Path to the saved Keras model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    if config is None:
        config = load_config()
    
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    
    acc = accuracy_score(y_true, y_pred_classes)
    report = classification_report(y_true, y_pred_classes)
    
    print(f'Accuracy: {acc:.4f}')
    print('Classification Report:')
    print(report)
