# code_snippets/scikit_learn/evaluation.py

from sklearn.metrics import accuracy_score, classification_report

def evaluate_sklearn_model(model, X_test, y_test):
    """
    Evaluate the scikit-learn model on test data.

    Args:
        model: Trained scikit-learn model.
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
