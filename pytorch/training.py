# code_snippets/pytorch/training.py

import torch

def train_pytorch_model(model, train_loader, criterion, optimizer, device, epochs=10, val_loader=None, early_stopping_patience=5):
    """
    Train the PyTorch model.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device (torch.device): Device to train on.
        epochs (int): Number of epochs.
        val_loader (DataLoader, optional): Validation data loader.
        early_stopping_patience (int): Early stopping patience.

    Returns:
        model (torch.nn.Module): Trained model.
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_loader.dataset)
            print(f'Epoch {epoch}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_pytorch_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch}')
                    model.load_state_dict(torch.load('best_pytorch_model.pth'))
                    break
        else:
            print(f'Epoch {epoch}/{epochs}, Train Loss: {epoch_loss:.4f}')

    return model
