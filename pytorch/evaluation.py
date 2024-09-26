# code_snippets/pytorch/evaluation.py

import torch
from sklearn.metrics import classification_report, accuracy_score
from utilities.common_functions import load_config

def evaluate_pytorch_model(model, test_loader, criterion, device='cpu'):
    """
    Evaluate the PyTorch model on test data.

    Args:
        model (torch.nn.Module): Model to evaluate.
        test_loader (DataLoader): Test data loader.
        criterion: Loss function.
        device (torch.device): Device.

    Returns:
        None
    """
    model.to(device)
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = batch_y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    test_loss /= len(test_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print('Classification Report:')
    print(report)
