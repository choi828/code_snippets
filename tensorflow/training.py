# code_snippets/tensorflow/training.py

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utilities.common_functions import load_config

def compile_and_train_tensorflow_model(model, X_train, y_train, X_val, y_val, config=None):
    """
    Compile and train the TensorFlow model.

    Args:
        model (tf.keras.Model): TensorFlow model to train.
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation labels.
        config (dict): Configuration dictionary.

    Returns:
        history (tf.keras.callbacks.History): Training history.
    """
    if config is None:
        config = load_config()
    
    optimizer = config['training']['optimizer']
    loss = config['training']['loss']
    metrics = config['training']['metrics']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    early_stopping_patience = config['training']['early_stopping_patience']
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True),
        ModelCheckpoint('best_tensorflow_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return history
