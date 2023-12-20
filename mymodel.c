#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sodium.h>
#include "mymodel.h"

double sigmoid(double x) 
{
    return 1.0 / (1.0 + exp(-x));
}

double random_double(double min, double max) 
{
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

void ReadFile(int max_rows, int max_cols, double data[max_rows][max_cols], char* filename) 
{
    // Open the file in read mode
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error: Could not open the file.\n");
        exit(1);
    }

    int row = 0;
    int col = 0;

    while (fscanf(file, "%lf", &data[row][col]) == 1)
    {
        col++;
        char ch = fgetc(file);
        if (ch == '\n' || ch == EOF)
        {
            row++;
            col = 0;
        }
        if (col >= max_cols)
        {
            printf("Error: The file given does not have the same number of inputs/outputs as specified\n");
            exit(1);
        }
        if (row >= max_rows)
        {
            printf("\nSuccessfully read the input file: %s\n\n", filename);
            break;
        }
    }

    // Close the file
    fclose(file);
}

void OrganizeData(int num_train, int num_inputs, int num_outputs, int num_val, int max_rows, int max_cols,
                    double data[max_rows][max_cols], double X_train[num_train][num_inputs], double Y_train[num_train][num_outputs],
                    double X_val[num_val][num_inputs], double Y_val[num_val][num_outputs])
{
    for (int row = 0; row < num_train; row++)
    {
        for (int col = 0; col < num_inputs; col++)
        {
            X_train[row][col] = data[row][col];
        }
        for (int col = 0; col < num_outputs; col++)
        {
            Y_train[row][col] = data[row][col + num_inputs];
        }
    }

    for (int row = num_train; row < max_rows; row++)
    {
        for (int col = 0; col < num_inputs; col++)
        {
            X_val[row - num_train][col] = data[row][col];
        }
        for (int col = 0; col < num_outputs; col++)
        {
            Y_val[row - num_train][col] = data[row][col + num_inputs];
        }
    }
}

void InitializeArrays(int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                        int num_train, double initial_range,
                        double ***W, double **b, double ***a)
{
    // Initialize weights
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

    // Initialize biases
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        b[layer] = malloc(num_neurons_current_layer * sizeof(double));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            b[layer][neuron] = random_double(-initial_range, initial_range);
        }
    }

    // Initialize activations
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        a[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            a[layer][neuron] = malloc(num_train * sizeof(double));
        }
    }
}

void CalculateMetrics(int num_train, int num_val, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                        double X_train[num_train][num_inputs], double Y_train[num_train][num_outputs], 
                        double X_val[num_val][num_inputs], double Y_val[num_val][num_outputs],
                        double ***W, double **b, 
                        double *accuracy_train, double *accuracy_val, double *cost_train, double *cost_val)
{

    // Allocate memory for training network activations
    double ***a_train = malloc((num_hidden_layers + 1) * sizeof(double **));
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        a_train[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            a_train[layer][neuron] = malloc(num_train * sizeof(double));
        }
    }

    ForwardPass(num_train, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                X_train,
                W, b, a_train);

    // Allocate memory for output neurons
    double **output_neurons_train = malloc(num_train * sizeof(double *));
    for (int i = 0; i < num_train; i++) {
        output_neurons_train[i] = malloc(num_outputs * sizeof(double));
    }

    // Transpose a_train[num_hidden_layers] to output_neurons_train
    for (int i = 0; i < num_outputs; i++) {
        for (int j = 0; j < num_train; j++) {
            output_neurons_train[j][i] = a_train[num_hidden_layers][i][j];
        }
    }

    // Calculate cost
    double sum_squared_diff = 0.0;
    for (int i = 0; i < num_train; i++) {
        for (int j = 0; j < num_outputs; j++) {
            double diff = output_neurons_train[i][j] - Y_train[i][j];
            sum_squared_diff += diff * diff;
        }
    }
    *cost_train = sum_squared_diff / num_train;

    // Calculate binary accuracy
    int correct_predictions = 0;
    for (int i = 0; i < num_train; i++) {
        int all_correct = 1;
        for (int j = 0; j < num_outputs; j++) {
            if ((output_neurons_train[i][j] >= 0.5 && Y_train[i][j] == 0) || (output_neurons_train[i][j] < 0.5 && Y_train[i][j] == 1)) {
                all_correct = 0;
                break;
            }
        }
        correct_predictions += all_correct;
    }
    *accuracy_train = (double)correct_predictions / num_train;

    // Allocate memory for validation network activations
    double ***a_val = malloc((num_hidden_layers + 1) * sizeof(double **));
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        a_val[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            a_val[layer][neuron] = malloc(num_val * sizeof(double));
        }
    }

    // Forward pass for validation network
    ForwardPass(num_val, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                X_val,
                W, b, a_val);

    // Allocate memory for output neurons
    double **output_neurons_val = malloc(num_val * sizeof(double *));
    for (int i = 0; i < num_val; i++) {
        output_neurons_val[i] = malloc(num_outputs * sizeof(double));
    }

    // Transpose a_val[num_hidden_layers] to output_neurons_val
    for (int i = 0; i < num_outputs; i++) {
        for (int j = 0; j < num_val; j++) {
            output_neurons_val[j][i] = a_val[num_hidden_layers][i][j];
        }
    }

    // Calculate cost
    sum_squared_diff = 0.0;
    for (int i = 0; i < num_val; i++) {
        for (int j = 0; j < num_outputs; j++) {
            double diff = output_neurons_val[i][j] - Y_val[i][j];
            sum_squared_diff += diff * diff;
        }
    }
    *cost_val = sum_squared_diff / num_val;

    correct_predictions = 0;
    for (int i = 0; i < num_val; i++) {
        int all_correct = 1;
        for (int j = 0; j < num_outputs; j++) {
            if ((output_neurons_val[i][j] >= 0.5 && Y_val[i][j] == 0) || (output_neurons_val[i][j] < 0.5 && Y_val[i][j] == 1)) {
                all_correct = 0;
                break;
            }
        }
        correct_predictions += all_correct;
    }
    *accuracy_val = (double)correct_predictions / num_val;

    // Free memory for a_train
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            free(a_train[layer][neuron]);
        }
        free(a_train[layer]);
    }
    free(a_train);

    // Free memory for output_neurons_train
    for (int i = 0; i < num_train; i++) {
        free(output_neurons_train[i]);
    }
    free(output_neurons_train);

    // Free memory for a_val
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            free(a_val[layer][neuron]);
        }
        free(a_val[layer]);
    }
    free(a_val);

    // Free memory for output_neurons_val
    for (int i = 0; i < num_val; i++) {
        free(output_neurons_val[i]);
    }
    free(output_neurons_val);
}

void ForwardPass(int num_train, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                double X_train[][num_inputs],
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
    double ***W = malloc((num_hidden_layers + 1) * sizeof(double **)); // Weights
    double **b = malloc((num_hidden_layers + 1) * sizeof(double *)); // Biases
    double ***a = malloc((num_hidden_layers + 1) * sizeof(double **)); // Activations

    // Initialize arrays
    InitializeArrays(num_inputs, num_outputs, num_hidden_layers, num_neurons, num_train, initial_range, W, b, a);

    for (int ep = 1; ep <= epochs; ep++) {

        ForwardPass(num_train, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                    X_train,
                    W, b, a);
        
        BackwardPass(learning_rate, num_train, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                    X_train, Y_train,
                    W, b, a);

        // Evaluate ANN every 100 epochs
        if (ep % 100 == 0) {
            double cost_train, cost_val, accuracy_train, accuracy_val;

            CalculateMetrics(num_train, num_val, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                            X_train, Y_train, X_val, Y_val,
                            W, b,
                            &accuracy_train, &accuracy_val, &cost_train, &cost_val);

            printf("Epoch %d:\n", ep);
            printf("Train Cost:      %lf     Accuracy: %.2f%%\n", cost_train, accuracy_train * 100);
            printf("Validation Cost: %lf     Accuracy: %.2f%%\n\n", cost_val, accuracy_val * 100);
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