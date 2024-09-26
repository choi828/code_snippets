# scripts/keras_train_example.py

import numpy as np
from keras.data_preprocessing import load_data, preprocess_data, split_data
from keras.model_architectures import create_keras_model
from keras.training import compile_and_train_keras_model
from keras.evaluation import evaluate_keras_model
from sklearn.preprocessing import OneHotEncoder

# 데이터 로드
X, y = load_data()

# 데이터 전처리
X_scaled, y_encoded, scaler, encoder = preprocess_data(X, y)

# 데이터 분할
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y_encoded)

# 모델 생성
model = create_keras_model(
    input_dim=X_train.shape[1],
    output_dim=y_train.shape[1]
)

# 모델 학습
history = compile_and_train_keras_model(
    model, X_train, y_train, X_val, y_val
)

# 모델 평가
evaluate_keras_model('best_keras_model.h5', X_test, y_test)
