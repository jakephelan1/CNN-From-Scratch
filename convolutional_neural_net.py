import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from scipy.signal import correlate2d
import joblib
from layers import Dense, Convolutional, Flatten, Dropout, MaxPooling
from activations import ReLU, Softmax
from losses import categorical_cross_entropy, categorical_cross_entropy_prime
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

def preprocess_data(x, y):
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype('float32') / 255
    y = to_categorical(y, 10)  
    y = y.reshape(len(y), 10)
    return x, y

def create_network():
    return [
        Convolutional((1, 28, 28), 3, 32),
        ReLU(),
        MaxPooling(2, 2),
        Convolutional((32, 13, 13), 3, 64),
        ReLU(),
        MaxPooling(2, 2),
        Flatten(),
        Dense(64 * 5 * 5, 128),
        ReLU(),
        Dropout(0.3),
        Dense(128, 10),
        Softmax()
    ]

def create_facial_recognition_network(num_classes):
    return [
        Convolutional((3, 224, 224), 3, 64),
        ReLU(),
        MaxPooling(2, 2),
        Convolutional((64, 111, 111), 3, 128),
        ReLU(),
        MaxPooling(2, 2),
        Convolutional((128, 55, 55), 3, 256),
        ReLU(),
        Convolutional((256, 55, 55), 3, 256),
        ReLU(),
        MaxPooling(2, 2),
        Convolutional((256, 27, 27), 3, 512),
        ReLU(),
        Convolutional((512, 27, 27), 3, 512),
        ReLU(),
        MaxPooling(2, 2),
        Flatten(),
        Dense(512 * 13 * 13, 4096),
        ReLU(),
        Dropout(0.5),
        Dense(4096, 4096),
        ReLU(),
        Dropout(0.5),
        Dense(4096, num_classes),
        Softmax()
    ]

def clip_and_apply_gradients(network, learning_rate, max_norm=1.0):
    gradients = []
    for layer in network:
        if isinstance(layer, Convolutional):
            gradients.extend([layer.kernels_gradient, layer.biases_gradient])
        elif isinstance(layer, Dense):
            gradients.extend([layer.weights_gradient, layer.bias_gradient])

    total_norm = np.sqrt(sum(np.sum(np.square(grad)) for grad in gradients if grad is not None))
    clip_coef = min(max_norm / total_norm + 1e-6, 1.0) if total_norm > max_norm else 1.0

    for layer in network:
        if hasattr(layer, 'apply_gradients'):
            layer.apply_gradients(learning_rate * clip_coef)

def reduce_lr_on_plateau(current_lr, val_acc_history, factor=0.5, patience=3, min_lr=1e-6):
    if len(val_acc_history) <= patience:
        return current_lr
    
    if all(val_acc_history[-i-1] >= val_acc_history[-i] for i in range(1, patience + 1)):
        new_lr = max(current_lr * factor, min_lr)
        logging.info(f"Validation accuracy not improving. Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")
        return new_lr
    
    return current_lr

def predict(network, x_batch):
    for layer in network:
        if hasattr(layer, 'set_training_mode') and layer.training == True:
            layer.set_training_mode(False)
    
    output = x_batch
    for layer in network:
        output = layer.forward(output)
    return np.argmax(output, axis=1)

def evaluate(network, x_test, y_test, batch_size):
    correct_predictions = 0
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i:i + batch_size]
        y_batch = y_test[i:i + batch_size]
        predictions = predict(network, x_batch)
        correct_predictions += np.sum(predictions == np.argmax(y_batch, axis=1))
    accuracy = correct_predictions / len(x_test)
    logging.info(f"Accuracy on test set: {accuracy * 100:.2f}%")


def train(network, x_train, y_train, epochs, batch_size, learning_rate, x_val, y_val):
    error_history = []
    val_acc_history = []
    
    for e in range(epochs):
        epoch_error = 0
        batch_count = 0
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
    
            output = x_batch
            for layer in network:
                if hasattr(layer, 'set_training_mode') and layer.training == False:
                    layer.set_training_mode(True)
                output = layer.forward(output)
     
            batch_error = categorical_cross_entropy(y_batch, output)
            epoch_error += batch_error
            batch_count += 1
  
            grad = categorical_cross_entropy_prime(y_batch, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
            
            clip_and_apply_gradients(network, learning_rate)
            
            print(f"Epoch {e + 1}/{epochs}, Batch {batch_count}, Error: {batch_error:.6f}")
        
        average_epoch_error = epoch_error / batch_count
        error_history.append(average_epoch_error)

        val_predictions = predict(network, x_val)
        val_accuracy = np.mean(val_predictions == np.argmax(y_val, axis=1))
        val_acc_history.append(val_accuracy)

        logging.info(f"Epoch {e + 1}/{epochs}, Average Error: {average_epoch_error:.6f}, Validation Accuracy: {val_accuracy:.4f}, lr = {learning_rate:.6f}")
        
        if np.isnan(average_epoch_error):
            print("NaN error encountered. Stopping training.")
            break

        learning_rate = reduce_lr_on_plateau(learning_rate, val_acc_history)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
x_test, y_test = preprocess_data(x_test, y_test)

epochs = 20  
batch_size = 128
learning_rate = 0.1
network = create_network()

train(network, x_train, y_train, epochs, batch_size, learning_rate, x_val, y_val)
joblib.dump(network, "cnn_model.pkl")

evaluate(network, x_test, y_test, batch_size)




