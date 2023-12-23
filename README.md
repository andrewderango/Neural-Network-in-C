# Developing an Artificial Neural Network in C

## Project Overview
This project focuses on the development of an Artificial Neural Network (ANN) entirely from scratch using the C programming language. At its core, the program is designed to orchestrate the creation, training, and deployment of a feedforward neural network (FNN). FNNs, a foundational architecture in the realm of ANNs, are characterized by their ability to learn complex relationships for predicting outcomes based on independent variables using a network of interconnected nodes. This project aims to establish a robust framework for constructing and training ANNs tailored to user-provided datasets.

## Key Features
In the development of this program, one of the core principles that were adhered to was to maximize the flexibility of the application for the end user to ultimately provide them with a versatile tool. To achieve this objective, we implemented a variety of features to create a robust and adaptable platform that can cater to diverse user needs.

1. **Customizable Architecture**: Upon execution, users can specify the number of features, labels, hidden layers, and neurons within these hidden layers. They can also adjust the learning rate, epochs, and activation function to ensure that users can leverage the optimal network architecture for their dataset.
2. **Integrated Test/Train Split Mechanism**: Users can specify how much of the input data they would like to use for training the model, and how much of the data should be allocated to validation. This helps notify users if the model that they are training is overfitting by showing how well it performs on new data.
3. **Real-time Performance Metrics**: During training, the program will update the user on the performance of the model on both the training and validation datasets. Not only does this give the user an idea of the effectiveness of the model, but can show the user how many epochs are sufficient for the training of the model on their dataset.
4. **Regression and Classification Support**: The artificial neural network can make predictions for both regression and classification problems.
5. **Exporting Models**: After the model is trained, users can choose to export their model. The network architecture, weights and biases, will be exported to a new file if the user decides to download the model.
6. **Performance Metrics**: Several performance metrics can be used to evaluate the model's performance on both training and validation datasets. Regression and classification metrics such as accuracy, MSE, $R^2$, and ROC AUC are available for use.
