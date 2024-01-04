# Developing an Artificial Neural Network in C

## Project Overview
This project focuses on the development of an Artificial Neural Network (ANN) entirely from scratch using the C programming language. At its core, the program is designed to orchestrate the creation, training, and deployment of a feedforward neural network (FNN). FNNs, a foundational architecture in the realm of ANNs, are characterized by their ability to learn complex relationships for predicting outcomes based on independent variables using a network of interconnected nodes. This project aims to establish a robust framework for constructing and training ANNs tailored to user-provided datasets.

## Demo
A brief demo video that covers an example of training a simple network from the data.txt file is available [here](https://www.youtube.com/watch?v=I1-ug0xZipE).

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
 For example, to train an FNN using 100,000 epochs, 0.0001 learning rate, 10% of the data used for training, and architecture of 2-64-32-18-2, then run the following command:
```
./ANN 100000 0.0001 0.1 2 64 32 18 2
```

## Program Structure

```main``` is where the user’s execution command is extracted, and where all functions are called. It starts by ensuring that the user’s execution command follows proper formatting, then assigns their inputs to variables to be passed into future function calls. It prints the network’s architecture, and then asks for the file name of the input data. Then, it calls ```OrganizeData``` and ```Evaluation```.

```ReadFile``` reads the file specified by the user. It iterates through the file twice. It first counts the number of rows and then uses this value to dynamically allocate memory for ```**data```, a 2D array that stores the data from the input file. It then reads through the file again and fills the array with the file data now that the array is declared. The function returns both this ```**data``` array and the number of rows in the file.

```OrganizeData``` first shuffles every data point that is passed in via ```**data```. It then computes how many training and validation data points there should be, based on the train/validation split determined by the user upon execution. It then assigns the proper number of data points to the training and validation datasets.

```InitializeArrays``` simply iterates through the weights, biases, and activations arrays and allocates the necessary memory for each element within each array. For the weights and biases, it assigns a different random double between (```-initial_range```, ```initial_range```) to each of them.

```CalculateMetrics``` computes the metrics that are printed every hundredth epoch during the model’s training so that the user can get a gauge of the ANN’s performance throughout training. This includes a variation of MSE and accuracy. To do this, it creates alternative activation 3D arrays and runs ```ForwardPass``` for both the training and validation datasets to assign activations, and further, predictions for the model on each dataset. For more information, read the dynamic memory allocation section which delves deeper into the memory handling and logistics of this function.

```ForwardPass``` assigns values for each training example to each neuron in each layer via the ```***a``` array. The three dimensions arise from ```a[layer][neuron][training_example]```. The activations for the first layer are simply the inputs from the data file. The activations of each of the following layers must be computed sequentially via a linear combination of the previous layer’s activation and the weights, bias, and activation function associated with the neuron in question, hence the function name “ForwardPass” and the type of model “feedforward neural network”.

The ```BackwardPass``` function deals with the backpropagation algorithm responsible for optimizing the weights and biases selected by the program to best predict the label values. The weights and biases are updated in each epoch, a change dependent on their influences on the ultimate predictions, and by extension, the cost function. It starts by allocating memory in all three dimensions for ```PL```, which stores the partial derivative of the cost function with respect to the activations of the neurons. This process begins by taking the derivative of the cost function with respect to the activations of the neurons in the output layer, which is simply the derivative of the activation function. The activation function is the sigmoid function and was computed manually to be in the form seen in the double ```for``` loop. Then, the function enters a loop for each hidden layer and computes ```PL``` for each iteration. Since ```PL``` is a partial derivative, we can split it into the intermediate partial derivatives via the chain rule, and calculate it as the derivative of the cost function, computed manually, multiplied by the weighted sum of the gradients of the neurons in the following layer of the network. Hence the function name “BackwardPass” and the name of the algorithm “backpropagation”. Using the gradients computed in the previous loop and the learning rate specified by the user, we update the weight of each synapse and the bias of each neuron.

The ```Evaluation``` function calls ```InitializeArrays``` then loops through the epochs of training, calling ```ForwardPass``` and ```BackwardPass``` each iteration. If the iteration is a hundredth epoch, then ```CalculateMetrics``` is called to show the performance. Following training, ```DownloadANN``` is called.

```DownloadANN``` is the function responsible for downloading or saving the network that was just trained. It first asks the user if they would like to save the network, then creates a directory called Downloaded Data to their current directory if they say yes and the directory does not already exist. It creates the file ```ANN_data.txt``` in this folder and then writes the model’s architecture, performance metrics, and weights and biases into it.
