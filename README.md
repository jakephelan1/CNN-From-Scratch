# CNN-From-Scratch

## Project Overview

This project involves building a Convolutional Neural Network (CNN) from scratch using Python and popular deep learning libraries. The aim is to demonstrate a comprehensive understanding of neural networks by implementing and training a CNN model to classify images. The project involves several complexities, including data preprocessing, model architecture design, training the model, and evaluating its performance.

## Project Structure

- `activations.py`: Contains activation functions and activation layer classes.
- `convolutional_neural_net.py`: Main script for defining, training, and evaluating the CNN model.
- `layers.py`: Contains the different layer classes used in the CNN, including Convolutional, Dense, Flatten, Dropout, and MaxPooling layers.
- `losses.py`: Contains loss functions and their derivatives.
- `training.log`: Log file for recording training progress and validation accuracy.
- `README.md`: Project documentation.

## Setup and Installation

### Prerequisites

Ensure you have Python 3.8 or later installed on your system. You can check your Python version by running:

```bash
python --version
```

### Installation 
1. **Clone the repository:** 
```bash
git clone https://github.com/jakephelan1/CNN-From-Scratch.git cd CNN-From-Scratch` 
```
2. **Create a virtual environment:** 
```bash
python -m venv venv source venv/bin/activate # On Windows, use venv\Scripts\activate
```
3. **Install the required packages:** 
```bash
pip install -r requirements.txt
``` 

### Data Preparation 
As of now, the program is set to train the model on the keras mnist dataset. If youd like to use custom data, ensure your dataset is placed in a `data/` directory and update the paths in the `preprocess_data` function in `convolutional_neural_net.py` accordingly. 

### Training/Testing the Model
 Run the following command to train and test the CNN model on the data: 
```bash
python convolutional_neural_network.py
```

This will start the training process, save the trained model as `cnn_model.pkl`, and evaluate its performance on a test set.





