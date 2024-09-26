# code_snippets/tensorflow/model_architectures.py

import tensorflow as tf
from utilities.common_functions import load_config

def create_tensorflow_model(input_dim, output_dim, hidden_units=[64, 128], dropout_rates=[0.5, 0.5], activation='relu', output_activation='softmax'):
    """
    Create a simple TensorFlow Keras Sequential model.

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output classes.
        hidden_units (list): Number of units in each hidden layer.
        dropout_rates (list): Dropout rates after each hidden layer.
        activation (str): Activation function for hidden layers.
        output_activation (str): Activation function for output layer.

    Returns:
        model (tf.keras.Model): TensorFlow model.
    """
    model = tf.keras.Sequential()
    for i, (units, dropout) in enumerate(zip(hidden_units, dropout_rates)):
        if i == 0:
            model.add(tf.keras.layers.Dense(units, activation=activation, input_shape=(input_dim,)))
        else:
            model.add(tf.keras.layers.Dense(units, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(output_dim, activation=output_activation))
    return model
