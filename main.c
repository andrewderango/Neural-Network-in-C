#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sodium.h>
#include "mymodel.h"

#define INITIAL_RANGE 0.2
#define MAX_ROWS 48120
#define MAX_COLS 4

int main(int argc, char *argv[]) {
    if (argc < 7) {
        printf("Sorry, I could not understand your input. Please refer to the following execution format:\n");
        printf("./ANN <epochs> <learning_rate> <train_split> <num_inputs> <num_neurons_layer2> ... <num_neurons_layerN> <num_outputs>\n");
        return 1;
    }

    int epochs = atoi(argv[1]);
    if (epochs < 1) {
        printf("Error: epochs parameter must be >= 1\n");
        return 1;
    }

    double learning_rate = atof(argv[2]);
    if (learning_rate <= 0) {
        printf("Error: learning_rate parameter must be > 0\n");
        return 1;
    }

    double train_split = atof(argv[3]);
    if (train_split <= 0 || train_split > 1) {
        printf("Error: train_split parameter must be > 0 and <= 1\n");
        return 1;
    }

    int num_inputs = atoi(argv[4]);
    if (num_inputs < 1) {
        printf("Error: num_inputs parameter must be >= 1\n");
        return 1;
    }

    int num_hidden_layers = argc - 6;
    int *num_neurons = malloc(num_hidden_layers * sizeof(int));
    for (int i = 0; i < num_hidden_layers; i++) {
        num_neurons[i] = atoi(argv[i + 5]);
        if (num_neurons[i] < 1) {
            printf("Error: num_neurons parameter for hidden layer %d must be >= 1\n", i + 1);
            return 1;
        }
    }

    int num_outputs = atoi(argv[argc - 1]);
    if (num_outputs < 1) {
        printf("Error: num_outputs parameter must be >= 1\n");
        return 1;
    }

    // Printing out user inputs
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
    printf("\n");

    // Initialize the 2D array to store the data
    double data[MAX_ROWS][num_inputs + num_outputs];
    char *filename = "data.txt";

    // There should be a better way to handle these define elements
    int max_rows = MAX_ROWS;
    int max_cols = num_inputs + num_outputs; // remove the #define above?????????????????????????????????
    double init_range = INITIAL_RANGE;
    
    ReadFile(max_rows, max_cols, data, filename);

    int num_train = MAX_ROWS * train_split + 1;
    int num_val = MAX_ROWS * (1 - train_split);

    double X_train[num_train][num_inputs];
    double Y_train[num_train][num_outputs];
    double X_val[num_val][num_inputs];
    double Y_val[num_val][num_outputs];

    OrganizeData(num_train, num_inputs, num_outputs, num_val, max_rows, max_cols,
                 data, X_train, Y_train, X_val, Y_val);

    Evaluation(num_inputs, num_outputs, num_hidden_layers, num_neurons, 
               epochs, learning_rate, init_range, num_train, num_val,
               X_train, Y_train, X_val, Y_val);

    free(num_neurons);
    return 0;
}