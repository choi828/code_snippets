# scripts/pytorch_train_example.py

import numpy as np
import torch
from pytorch.data_preprocessing import load_data, preprocess_data, split_data
from pytorch.model_architectures import SimplePyTorchModel
from pytorch.training import train_pytorch_model
from pytorch.evaluation import evaluate_pytorch_model

# 데이터 로드
X, y = load_data()

# 데이터 전처리
X_scaled, y_labels, scaler = preprocess_data(X, y)

# 데이터 분할
train_dataset, val_dataset, test_dataset = split_data(X_scaled, y_labels)

# DataLoader 생성
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# 모델 생성
model = SimplePyTorchModel(
    input_dim=X_scaled.shape[1],
    output_dim=10
)

# 손실 함수 및 옵티마이저 설정
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 학습
train_pytorch_model(
    model, train_loader, criterion, optimizer, 
    device=device,
    epochs=20, 
    val_loader=val_loader,
    early_stopping_patience=5
)

# 모델 평가
evaluate_pytorch_model(
    model, test_loader, criterion, 
    device=device
)
