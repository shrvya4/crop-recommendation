
# SoilToSuccess

A machine learning-based crop recommendation system designed to suggest suitable crops based on various factors such as soil type, climate conditions, and location. This system aims to optimize agricultural production by providing farmers with data-driven insights for crop selection.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Model](#model)
- [Contributing](#contributing)


## Overview

The Crop Recommender System helps farmers and agricultural enthusiasts select the best crops based on several factors. The system uses data like:

- **Soil Properties** (pH, texture, fertility)
- **Climate Conditions** (temperature, humidity)
- **Geographic Location**
- **Seasonal Patterns**

It leverages a **Neural Network** model to predict the most suitable crops for a given set of conditions, with **hyperparameter tuning** using **Grid Search** to optimize model performance.

## Features

- **Crop Recommendation**: Recommends crops based on soil and climate data.
- **User Input**: Allows users to input their soil and climate conditions to get personalized crop recommendations.
- **Data Analysis**: Utilizes statistical analysis to assess various factors influencing crop growth.
- **Optimized Model**: Hyperparameter tuning with Grid Search for better accuracy.
- **Interactive Interface**: Provides an easy-to-use interface for input and result display.

## Technologies Used

- **Python**: The primary programming language for building the system.
- **Keras/TensorFlow**: For building and training the neural network model.
- **Scikit-Learn**: For machine learning utilities like Grid Search and data preprocessing.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib/Seaborn**: For visualizing data and results.
- **Flask/Django (Optional)**: For creating a web-based interface.

## Setup and Installation

### Prerequisites

- Python 3.6+
- pip (Python package installer)

## Usage

1. **Input**: The system requires the user to provide information about:
   - Soil properties (e.g., pH, texture,salinity, organic matter)
   - Climate conditions (e.g., temperature)
   - Location (optional for localized recommendations)
   
2. **Output**: Based on the input, the system will return a list of recommended crops that would be suitable for the given conditions.

## Data Sources

The model is trained on various agricultural datasets,

- [Soil Quality Data](https://example.com/soil_data)
- [Climate Data](https://example.com/climate_data)
- [Crop Yield Data](https://example.com/crop_yield_data)

> Note: You can modify the dataset sources according to your own datasets.

## Model

The system uses a **Neural Network** model to predict the best crop recommendations based on the given input features. The model is optimized using **Grid Search** to find the best hyperparameters for improved accuracy.

### Model Architecture

- **Input Layer**: Takes features such as soil properties, climate conditions, and location.
- **Hidden Layers**: Multiple fully connected layers with activation functions (e.g., ReLU) to learn complex relationships between inputs and output.
- **Output Layer**: Predicts the best crops (multi-class classification).

### Hyperparameter Tuning with Grid Search

To optimize the neural network model, **Grid Search** is used to search for the best combination of hyperparameters, such as:

- Number of hidden layers and units per layer.
- Learning rate.
- Batch size.
- Optimizer (e.g., Adam, SGD).

### Training the Model

To train the model:

1. **Prepare your dataset**: Ensure that the dataset is cleaned, preprocessed, and split into training and testing sets.
   
2. **Run Grid Search**: Execute the following command to perform hyperparameter tuning using Grid Search

3. **Train the Neural Network**: Once the best parameters are found, run the training script to train the model

4. The model will be saved in the `model/` directory as `crop_recommender_model.h5` for future use.

### Example code for Grid Search (in `grid_search.py`):

```python
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def create_model(learning_rate=0.001, num_layers=2, num_units=64):
    model = Sequential()
    model.add(Dense(num_units, input_dim=X_train.shape[1], activation='relu'))
    
    for _ in range(num_layers - 1):
        model.add(Dense(num_units, activation='relu'))
    
    model.add(Dense(y_train.shape[1], activation='softmax'))  # For multi-class classification
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Grid search for hyperparameters
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)
param_grid = {
    'learning_rate': [0.001, 0.01],
    'num_layers': [1, 2],
    'num_units': [64, 128]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Print best parameters
print(f"Best Parameters: {grid_result.best_params_}")
```

Once the best parameters are found, the final model is trained with the optimal configuration.

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, feel free to fork the repository, create a pull request, and submit your changes. Please ensure that you follow the coding standards and provide proper documentation.


To report issues, please use the Issues section


Sample Output:

![output](https://github.com/user-attachments/assets/db9490f1-40ba-42bc-97a3-27c0cd6968ef)


