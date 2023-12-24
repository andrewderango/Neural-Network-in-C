#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sodium.h>
#include "mymodel.h"

#define INITIAL_RANGE 0.2 // Initial range (positive and negative) for random weights and biases

int main(int argc, char *argv[]) {
    // User must enter at least 7 arguments in the command line to define network architecture
    if (argc < 7) {
        printf("Sorry, I could not understand your input. Please refer to the following execution format:\n");
        printf("./ANN <epochs> <learning_rate> <train_split> <num_inputs> <num_neurons_layer2> ... <num_neurons_layerN> <num_outputs>\n");
        return 1;
    }

    // Extract the epochs to train the model with
    int epochs = atoi(argv[1]);
    if (epochs < 1) {
        printf("Error: The number of epochs must be at least 1.\n");
        return 1;
    }

    // Extract the learning rate at which the model will be trained with
    double learning_rate = atof(argv[2]);
    if (learning_rate <= 0) {
        printf("Error: The learning rate must be greater than 0.\n");
        return 1;
    }

    // Extract the proportion of the data to be used for training. The rest will be used for validation
    double train_split = atof(argv[3]);
    if (train_split <= 0 || train_split > 1) {
        printf("Error: The training split must be a proportion of the input data, between 0 (exclusive) and 1 (inclusive).\n");
        return 1;
    }

    // Extract the number of variables that are used as inputs to the model (independent variables)
    int num_inputs = atoi(argv[4]);
    if (num_inputs < 1) {
        printf("Error: The number of features must be at least 1.\n");
        return 1;
    }

    // Extract the number of neurons in each hidden layer, in order
    int num_hidden_layers = argc - 6;
    int *num_neurons = malloc(num_hidden_layers * sizeof(int));
    // Iterate through each hidden layer and extract the number of neurons in each
    for (int i = 0; i < num_hidden_layers; i++) {
        num_neurons[i] = atoi(argv[i + 5]);
        if (num_neurons[i] < 1) {
            printf("Error: The quantity of neurons in hidden layer %d must be at least 1.\n", i + 1);
            return 1;
        }
    }

    // Extract the quantity of variables that are to be used as outputs for the model (dependent variables)
    int num_outputs = atoi(argv[argc - 1]);
    if (num_outputs < 1) {
        printf("Error: The number of labels must be at least 1.\n");
        return 1;
    }

    // Printing out user inputs prior to training
    printf("\n-- NEURAL NETWORK ARCHITECTURE --\n");
    printf("Epochs: %d\n", epochs);
    printf("Learning Rate: %f\n", learning_rate);
    printf("Train Split (Proportion): %f\n", train_split);
    printf("Input Neurons: %d\n", num_inputs);
    printf("Output Neurons: %d\n", num_outputs);
    printf("Hidden Layers: %d\n", num_hidden_layers);
    printf("Quantity of Neurons in Sequential Hidden Layers: ");
    for (int i = 0; i < num_hidden_layers; i++) {
        printf("%d ", num_neurons[i]);
    } 
    printf("\n\n");

    // Declare variables representing some training and input data information
    int num_cols = num_inputs + num_outputs;
    double init_range = INITIAL_RANGE;
    InputData input_data = ReadFile(num_cols);
    double **data = input_data.data;
    int num_rows = input_data.num_rows;
    char *filename = input_data.filename;
    int num_train = num_rows * train_split + 1;
    int num_val = num_rows * (1 - train_split);

    // Declare arrays to store the training and validation datasets
    double **X_train = malloc(num_train * sizeof(double *));
    double **Y_train = malloc(num_train * sizeof(double *));
    double **X_val = malloc(num_val * sizeof(double *));
    double **Y_val = malloc(num_val * sizeof(double *));
    for (int i = 0; i < num_train; i++) {
        X_train[i] = malloc(num_inputs * sizeof(double));
    }
    for (int i = 0; i < num_train; i++) {
        Y_train[i] = malloc(num_outputs * sizeof(double));
    }
    for (int i = 0; i < num_val; i++) {
        X_val[i] = malloc(num_inputs * sizeof(double));
    }
    for (int i = 0; i < num_val; i++) {
        Y_val[i] = malloc(num_outputs * sizeof(double));
    }
                                                                    
    // Separate the data into training and validation datasets, proportion defined by the user upon execution
    OrganizeData(num_train, num_inputs, num_outputs, num_rows,
                 data, X_train, Y_train, X_val, Y_val);

    // Train the model and evaluate performance
    Evaluation(num_inputs, num_outputs, num_hidden_layers, num_neurons, filename,
               epochs, learning_rate, init_range, num_train, num_val, train_split,
               X_train, Y_train, X_val, Y_val);

    // Free all dynamically allocated memory
    free(num_neurons);

    for (int i = 0; i < num_rows; i++) {
        free(input_data.data[i]);
    }
    free(input_data.data);

    for (int i = 0; i < num_train; i++) {
        free(X_train[i]);
    }
    free(X_train);

    for (int i = 0; i < num_train; i++) {
        free(Y_train[i]);
    }
    free(Y_train);

    for (int i = 0; i < num_val; i++) {
        free(X_val[i]);
    }
    free(X_val);

    for (int i = 0; i < num_val; i++) {
        free(Y_val[i]);
    }
    free(Y_val);

    return 0;
}