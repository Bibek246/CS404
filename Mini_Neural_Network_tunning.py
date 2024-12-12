#Keras examples: https://keras.io/examples/ 
#Help from here: https://stackoverflow.com/questions/36952763/how-to-return-history-of-validation-loss-in-keras 

# Imports and warning suppression 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# Tensorflow and keras imports 
from tensorflow import keras 
# Data Wrangling imports 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report 
import numpy as np 
# Graphing 
import matplotlib.pyplot as plt 

# Read in our faults csv
df = pd.read_csv("faults.csv")
# List out what our faults are
faults = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

#-------------------------------------------
# Create our dataset - features
# Features - everything but the faults
features = df.drop(faults, axis=1)
# Outcomes - just the faults 
outcomes = df[faults]
# Lets print both columns and shape 
print(features.columns)
print(outcomes.columns)
print(features.shape)
print(outcomes.shape)

#--------------------
# Data preprocessing 
# Normalize everything 
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
outcomes = scaler.fit_transform(outcomes)

# Split into a training and test set 
train_x, test_x, train_y, test_y = train_test_split(features, outcomes, test_size=0.2)

#-----------------------
# Build models with different architectures
# Function to create models with varying architectures
def create_model(layer_config, activation_function="relu"):
    input_layer = keras.layers.Input(shape=(train_x.shape[-1],))
    
    # Add hidden layers based on configuration
    layers = []
    for nodes in layer_config:
        layers.append(keras.layers.Dense(nodes, activation=activation_function))
    
    output_layer = keras.layers.Dense(train_y.shape[-1], activation="softmax")
    
    # Build model with the configured layers
    model = keras.models.Sequential([input_layer] + layers + [output_layer])
    
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss_function = keras.losses.CategoricalCrossentropy()
    metric_list = [keras.metrics.CategoricalAccuracy()]
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metric_list)
    
    return model

# Configuration for the three architectures
configurations = [
    ([30, 15], "relu"),  # Model 1: 2 layers, 30 and 15 nodes, relu activation
    ([40, 20, 10], "relu"),  # Model 2: 3 layers, 40, 20, and 10 nodes, relu activation
    ([30, 15], "tanh")  # Model 3: 2 layers, 30 and 15 nodes, tanh activation
]

# Common configurations
epochs = 10
batch_size = 16
callback_function = keras.callbacks.EarlyStopping(patience=2, monitor="val_loss", restore_best_weights=True)

# Store accuracy results
results = []

# Train and evaluate each architecture
for config in configurations:
    layer_config, activation = config
    model = create_model(layer_config, activation)
    
    # Train the model
    history = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, callbacks=[callback_function], validation_split=0.1, verbose=0)
    
    # Evaluate the model
    train_eval = model.evaluate(x=train_x, y=train_y, verbose=0)
    test_eval = model.evaluate(x=test_x, y=test_y, verbose=0)
    
    # Collect results
    results.append({
        "layers": layer_config,
        "activation": activation,
        "training_accuracy": train_eval[1],
        "test_accuracy": test_eval[1],
        "validation_accuracy": history.history["val_categorical_accuracy"][-1] if "val_categorical_accuracy" in history.history else None
    })

# Convert results to a DataFrame for easy viewing
results_df = pd.DataFrame(results)
print(results_df)

# Example of plotting training and validation accuracy for one model
accuracy_traces = ["categorical_accuracy", "val_categorical_accuracy"]
for trace in accuracy_traces:
    plt.plot(history.history[trace])
plt.legend(accuracy_traces)
plt.show()

# Save the final model in the recommended Keras format
model.save("my_model.keras")
