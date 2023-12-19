#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sodium.h>
#include "mymodel.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double random_double(double min, double max) {
    unsigned char buffer[sizeof(uint64_t)]; // Buffer to hold random bytes
    uint64_t random_value;

    // Initialize the sodium library
    if (sodium_init() < 0) {
        printf("Error initializing the sodium library.\n");
        exit(1);
    }

    // Generate random bytes
    randombytes_buf(buffer, sizeof(uint64_t));

    // Convert the random bytes to a 64-bit unsigned integer value
    memcpy(&random_value, buffer, sizeof(uint64_t));

    // Scale the 64-bit random integer to a double value in the range [0, 1.0]
    double scale = random_value / ((double)UINT64_MAX);

    // Scale the value to the desired range [min, max]
    return min + scale * (max - min);
}

void ReadFile(int MAX_ROWS, int MAX_COLS, double data[MAX_ROWS][MAX_COLS], char* filename)
{
    // Open the file in read mode
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error opening the file.\n");
        exit(1);
    }

    int row = 0;
    char line[100];

    // Read each line of the file
    while (fgets(line, sizeof(line), file))
    {
        // Tokenize the line and store the values in the array
        sscanf(line, "%lf %lf %lf %lf",
               &data[row][0], &data[row][1], &data[row][2], &data[row][3]);

        row++;
        // Break the loop if the array is full to prevent overflow
        if (row >= MAX_ROWS)
        {
            printf("\nSuccessfully read the input file: %s\n\n", filename);
            break;
        }
    }

    // Close the file
    fclose(file);
}

void ForwardPass(int num_train, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                double X_train[][num_inputs], double Y_train[][num_outputs],
                double ***W, double **b, double ***a)
{
    for (int layer = 0; layer < num_hidden_layers + 1; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        int num_neurons_previous_layer = (layer == 0) ? num_inputs : num_neurons[layer - 1];

        for (int i = 0; i < num_neurons_current_layer; i++) {
            for (int j = 0; j < num_train; j++) {
                double sum = 0;
                for (int k = 0; k < num_neurons_previous_layer; k++) {
                    sum += W[layer][i][k] * ((layer == 0) ? X_train[j][k] : a[layer - 1][k][j]);
                }
                a[layer][i][j] = (layer == num_hidden_layers) ? sigmoid(sum + b[layer][i]) : tanh(sum + b[layer][i]);
            }
        }
    }
}

void BackwardPass(double Learning_rate, int num_train, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                double X_train[][num_inputs], double Y_train[][num_outputs],
                double ***W, double **b, double ***a)
{
    double ***PL = malloc((num_hidden_layers + 1) * sizeof(double **));
    for (int layer = 0; layer < num_hidden_layers + 1; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        PL[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            PL[layer][neuron] = malloc(num_train * sizeof(double));
        }
    }

    // Calculate PL for the output layer
    for (int i = 0; i < num_outputs; i++) {
        for (int j = 0; j < num_train; j++) {
            PL[num_hidden_layers][i][j] = (a[num_hidden_layers][i][j] - Y_train[j][i]) * (1 - a[num_hidden_layers][i][j]) * a[num_hidden_layers][i][j];
        }
    }

    // Calculate PL for the hidden layers
    for (int layer = num_hidden_layers - 1; layer >= 0; layer--) {
        int num_neurons_current_layer = num_neurons[layer];
        int num_neurons_next_layer = (layer == num_hidden_layers - 1) ? num_outputs : num_neurons[layer + 1];

        double *a_squared_complement = malloc(num_neurons_current_layer * num_train * sizeof(double));
        for (int i = 0; i < num_neurons_current_layer; i++) {
            for (int j = 0; j < num_train; j++) {
                a_squared_complement[i * num_train + j] = 1 - a[layer][i][j] * a[layer][i][j];
            }
        }

        double *W_PL = malloc(num_neurons_current_layer * num_train * sizeof(double));
        for (int i = 0; i < num_neurons_current_layer; i++) {
            for (int j = 0; j < num_train; j++) {
                double sum = 0.0;
                for (int k = 0; k < num_neurons_next_layer; k++) {
                    sum += W[layer + 1][k][i] * PL[layer + 1][k][j];
                }
                W_PL[i * num_train + j] = sum;
            }
        }

        for (int i = 0; i < num_neurons_current_layer; i++) {
            for (int j = 0; j < num_train; j++) {
                PL[layer][i][j] = a_squared_complement[i * num_train + j] * W_PL[i * num_train + j];
            }
        }

        free(a_squared_complement);
        free(W_PL);
    }

    // Update weights and biases using learning_rate and PL
    for (int layer = 0; layer < num_hidden_layers + 1; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        int num_neurons_previous_layer = (layer == 0) ? num_inputs : num_neurons[layer - 1];

        for (int i = 0; i < num_neurons_current_layer; i++) {
            for (int j = 0; j < num_neurons_previous_layer; j++) {
                double sum = 0.0;
                for (int k = 0; k < num_train; k++) {
                    sum += PL[layer][i][k] * ((layer == 0) ? X_train[k][j] : a[layer - 1][j][k]);
                }
                W[layer][i][j] -= Learning_rate * sum;
            }
        }

        for (int i = 0; i < num_neurons_current_layer; i++) {
            double sum = 0.0;
            for (int j = 0; j < num_train; j++) {
                sum += PL[layer][i][j];
            }
            b[layer][i] -= Learning_rate * sum;
        }
    }

    for (int layer = 0; layer < num_hidden_layers + 1; layer++) {
        for (int neuron = 0; neuron < num_neurons[layer]; neuron++) {
            free(PL[layer][neuron]);
        }
        free(PL[layer]);
    }
    free(PL);
}

void Evaluation(int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                int epochs, double learning_rate, double initial_range, int num_train, int num_val,
                double X_train[num_train][num_inputs], double Y_train[num_train][num_outputs], 
                double X_val[num_val][num_inputs], double Y_val[num_val][num_outputs])
{
    // Allocate memory for weights
    double ***W = malloc((num_hidden_layers + 1) * sizeof(double **));
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        int num_neurons_previous_layer = (layer == 0) ? num_inputs : num_neurons[layer - 1];
        W[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            W[layer][neuron] = malloc(num_neurons_previous_layer * sizeof(double));
            for (int weight = 0; weight < num_neurons_previous_layer; weight++) {
                W[layer][neuron][weight] = random_double(-initial_range, initial_range);
            }
        }
    }

    // Allocate memory for biases
    double **b = malloc((num_hidden_layers + 1) * sizeof(double *));
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        b[layer] = malloc(num_neurons_current_layer * sizeof(double));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            b[layer][neuron] = random_double(-initial_range, initial_range);
        }
    }

    // Allocate memory for activations
    double ***a = malloc((num_hidden_layers + 1) * sizeof(double **));
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        a[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            a[layer][neuron] = malloc(num_train * sizeof(double));
        }
    }

    // // Print W
    // for (int layer = 0; layer <= num_hidden_layers; layer++) {
    //     int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
    //     int num_neurons_previous_layer = (layer == 0) ? num_inputs : num_neurons[layer - 1];
    //     printf("W[%d]:\n", layer);
    //     for (int i = 0; i < num_neurons_current_layer; i++) {
    //         for (int j = 0; j < num_neurons_previous_layer; j++) {
    //             printf("%lf ", W[layer][i][j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // // Print b
    // for (int layer = 0; layer <= num_hidden_layers; layer++) {
    //     int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
    //     printf("b[%d]:\n", layer);
    //     for (int i = 0; i < num_neurons_current_layer; i++) {
    //         printf("%lf ", b[layer][i]);
    //     }
    //     printf("\n\n");
    // }

    printf("Training...\n\n");

    for (int ep = 1; ep <= epochs; ep++) {

        ForwardPass(num_train, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                    X_train, Y_train,
                    W, b, a);
        
        BackwardPass(learning_rate, num_train, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                    X_train, Y_train,
                    W, b, a);

        if (ep % 100 == 0) {
            double train_cost = CalculateCost(num_train, num_outputs, Y_train, *a[num_hidden_layers]);
            double train_accuracy = CalculateAccuracy(num_train, num_outputs, Y_train, *a[num_hidden_layers]);
            double val_cost = CalculateCost(num_val, num_outputs, Y_val, *a[num_hidden_layers]);
            double val_accuracy = CalculateAccuracy(num_val, num_outputs, Y_val, *a[num_hidden_layers]);

            printf("Epoch %d:\n", ep);
            printf("Train Cost: %lf\n", train_cost);
            printf("Train Accuracy: %lf\n", train_accuracy);
            printf("Validation Cost: %lf\n", val_cost);
            printf("Validation Accuracy: %lf\n\n", val_accuracy);
        }
    }

    // Free the dynamically allocated memory for activations
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            free(a[layer][neuron]);
        }
        free(a[layer]);
    }
    free(a);

    for (int layer = 0; layer < num_hidden_layers; layer++) {
        free(W[layer]);
        free(b[layer]);
    }
    free(W);
    free(b);
}

double CalculateCost(int num_samples, int num_outputs, double Y[num_samples][num_outputs], double a[num_outputs]) {
    double cost = 0.0;
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_outputs; j++) {
            cost += (Y[i][j] - a[j]) * (Y[i][j] - a[j]);
        }
    }
    cost /= (2 * num_samples);
    return cost;
}

double CalculateAccuracy(int num_samples, int num_outputs, double Y[num_samples][num_outputs], double a[num_outputs]) {
    int correct_predictions = 0;
    for (int i = 0; i < num_samples; i++) {
        int predicted_class = 0;
        double max_activation = a[0];
        for (int j = 1; j < num_outputs; j++) {
            if (a[j] > max_activation) {
                predicted_class = j;
                max_activation = a[j];
            }
        }
        int true_class = 0;
        for (int j = 1; j < num_outputs; j++) {
            if (Y[i][j] > Y[i][true_class]) {
                true_class = j;
            }
        }
        if (predicted_class == true_class) {
            correct_predictions++;
        }
    }
    double accuracy = (double)correct_predictions / num_samples;
    return accuracy;
}