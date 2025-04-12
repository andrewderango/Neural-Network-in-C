# Developing an MLP Neural Network in C

## Project Overview
This project focuses on the development of an Artificial Neural Network (ANN) entirely from scratch using the C programming language. At its core, the program is designed to orchestrate the creation, training, and deployment of a multilayer perceptron (MLP) neural network. MLP neural networks, a foundational architecture in the realm of ANNs, are characterized by their ability to learn complex relationships for predicting outcomes based on independent variables using a network of interconnected nodes. This project aims to establish a robust framework for constructing and training ANNs tailored to user-provided datasets.

## Demo
A brief demo video that covers an example of training a simple network from the data.csv file is available [here](https://www.youtube.com/watch?v=I1-ug0xZipE).

## Features
In the development of this program, one of the core principles that were adhered to was to maximize the flexibility of the application for the end user to ultimately provide them with a versatile tool. To achieve this objective, we implemented a variety of features to create a robust and adaptable platform that can cater to diverse user needs.

1. **Customizable Architecture**: Upon execution, users can specify the number of features, labels, hidden layers, and neurons within these hidden layers. They can also adjust the learning rate, epochs, and activation function to ensure that users can leverage the optimal network architecture for their dataset.
2. **Integrated Test/Train Split Mechanism**: Users can specify how much of the input data they would like to use for training the model, and how much of the data should be allocated to validation. This helps notify users if the model that they are training is overfitting by showing how well it performs on new data.
3. **Real-time Performance Metrics**: During training, the program will update the user on the performance of the model on both the training and validation datasets. Not only does this give the user an idea of the effectiveness of the model, but can show the user how many epochs are sufficient for the training of the model on their dataset.
4. **Regression and Classification Support**: The artificial neural network can make predictions for both regression and classification problems.
5. **Exporting Models**: After the model is trained, users can choose to export their model. The network architecture, weights and biases, will be exported to a new file if the user decides to download the model.
6. **Performance Metrics**: Several performance metrics can be used to evaluate the model's performance on both training and validation datasets. Regression and classification metrics such as accuracy, MSE, R², binary cross-entropy, and ROC AUC are available for use.

<p align="center">
  <img src="https://github.com/andrewderango/Neural-Network-in-C/raw/main/Data%20Visualizations/metrics_vs_epoch.png" alt="Metrics vs Epoch" width="400">
  <img src="https://github.com/andrewderango/Neural-Network-in-C/raw/main/Data%20Visualizations/roc_curve.png" alt="ROC Curve" width="533">
</p>

## Using the ANN
### Dependencies
To run this program, the user must have a C compiler, the standard libraries, and the Libsodium library installed. Documentation and installation instructions can be found on their [official website](https://libsodium.gitbook.io/doc/).

#### Libsodium Installation on WSL/Linux

To install Libsodium on Windows Subsystem for Linux (WSL), follow these instructions:

1. Open your WSL/Linux terminal.
2. Update the package list:
```
sudo apt update
```
3. Install Libsodium:
```
sudo apt install libsodium-dev
```

#### Libsodium Installation on macOS

To install Libsodium on macOS, you can use Homebrew. If Homebrew is not installed, you can install it by following the instructions on the [Homebrew website](https://brew.sh/).

1. Open your terminal.
2. Install Libsodium using Homebrew:
```
brew install libsodium
```

After completing these steps, Libsodium should be installed on the user's system, and they may proceed to compile the program.

### Makefile Configuration
Upon downloading the program, it is necessary for the user to configure the Makefile. To seamlessly integrate the ```sodium.h``` library into the program, modifications to the Makefile are required, contingent on the operating system of the user's machine. It is imperative for the user to selectively activate only those sections in the Makefile that pertain to the active operating system during the code compilation process. The Makefile includes clearly labelled comments indicating the specific section to activate based on the operating system in use. The rest should be deleted or commented out.

### Running the Program
To run the program, follow the proceeding instructions:
1. Open your terminal.
2. Navigate to the directory that you would like to download the repository into.
3. Clone the repository: 
```
git clone https://github.com/andrewderango/Neural-Network-in-C
```
4. Enter the directory:
```
cd Neural-Network-in-C
```
5. Add the data file to this directory.
6. Ensure that the Makefile is configured properly, as mentioned in the Makefile Configuration section.
8. Compile the program: 
```
make
```
9. Execute the program using the following terminal command format:
```
./ANN <epochs> <learning_rate> <train_split> <num_inputs> <num_neurons_layer2> ... <num_neurons_layerN> <num_outputs>
```
 For example, to train an MLP neural network using 100,000 epochs, 0.0001 learning rate, 10% of the data used for training, and architecture of 2-64-32-18-2, then run the following command:
```
./ANN 100000 0.0001 0.1 2 64 32 18 2
```

## Sample Input Files
Within the repository, you'll find two CSV files: data.csv and data33.csv. These files are provided specifically for testing both the program and the neural network.

| File Name | Number of Rows | Input Variables | Output Variables | Input Variables Type | Output Variables Type | File Size |
|:---------:|:--------------:|:---------------:|:----------------:|:--------------------:|:---------------------:|:---------:|
| [data.csv](https://github.com/andrewderango/Neural-Network-in-C/blob/main/data.csv)  | 48,120         | 2             | 2              | ```float```          | ```bool```            | 1.5 MB    |
| [data33.csv](https://github.com/andrewderango/Neural-Network-in-C/blob/main/data33.csv)| 48,120         | 3             | 3              | ```float```          | ```bool```            | 2.2 MB    |

To enhance the understanding of the datasets, the following plots have been generated:

<p align="center">
  <img src="https://github.com/andrewderango/Neural-Network-in-C/assets/93727693/e9fea56c-a2f5-4835-9c7f-1be9d0080df6" width="750">
</p>
<p align="center">
  <img src="https://github.com/andrewderango/Neural-Network-in-C/assets/93727693/40b69df9-1d6c-46a4-b661-5a2b9da1a63f" width="750">
</p>

For reference, the ANN can achieve up to a 90% accuracy on these datasets!

This program is designed to seamlessly handle three input file extensions: .csv, .tsv, and .txt. It's important to emphasize that the .txt file is expected to adhere to the same formatting as the .tsv file for accurate parsing.