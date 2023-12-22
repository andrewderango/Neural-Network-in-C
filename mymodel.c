#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sodium.h>
#include <sys/stat.h>
#include "mymodel.h"

// Simple Sigmoid function for activation function
double Sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Generate a random double between min and max
double RandomDouble(double min, double max) {
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

// Read the data from the file into a 2D array, return the array and row qty
InputData ReadFile(char* filename, int num_cols) {
    // Open the file and read
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Sorry, I could not open %s. Please ensure that it is in the proper directory.\n", filename);
        exit(1);
    }

    // Count the number of rows in the file
    int num_rows = 0;
    double value_holder; // Garbage variable that holds the value read from the file
    while (fscanf(file, "%lf", &value_holder) == 1) {
        char next_character = fgetc(file);
        if (next_character == '\n' || next_character == EOF) {
            num_rows++;
        }
    }

    // Allocate memory for 2D array and fill with file data
    double** data = (double**)malloc(num_rows * sizeof(double *));
    for (int i = 0; i < num_rows; i++) {
        data[i] = (double*)malloc(num_cols * sizeof(double));
    }

    // Read the file again from beginning
    fseek(file, 0, SEEK_SET);

    // Transcribe the data from the file into the data array
    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols; col++) {
            fscanf(file, "%lf", &data[row][col]);
        }
    }

    // Close the file
    fclose(file);

    // Create and return DataResult struct
    InputData input_data;
    input_data.data = data;
    input_data.num_rows = num_rows;

    return input_data;
}

// Split the data into training and validation sets
void OrganizeData(int num_train, int num_inputs, int num_outputs, int num_rows,
                  double **data, double **X_train, double **Y_train, double **X_val, double **Y_val)
{
    // Array to store indices of each row of data
    int* datarow_indices = malloc(num_rows * sizeof(int));
    for (int i = 0; i < num_rows; i++) {
        datarow_indices[i] = i;
    }

    // Shuffle indices array randomly to randomize order of input data. For every row, pick random index and swap
    for (int i = 0; i < num_rows - 1; i++) {
        int j = (int)RandomDouble(i + 1, num_rows - 1);
        int prev_i_index = datarow_indices[i];
        datarow_indices[i] = datarow_indices[j];
        datarow_indices[j] = prev_i_index;
    }

    // Assign training rows to x and y training arrays for features and labels, respectively. Randomized order because of shuffled indices array
    for (int row = 0; row < num_train; row++) {
        int index = datarow_indices[row];
        for (int col = 0; col < num_inputs; col++) {
            X_train[row][col] = data[index][col];
        }
        for (int col = 0; col < num_outputs; col++) {
            Y_train[row][col] = data[index][col + num_inputs];
        }
    }

    // Assign validation rows to x and y validation arrays for features and labels, respectively
    for (int row = num_train; row < num_rows; row++) {
        int index = datarow_indices[row];
        for (int col = 0; col < num_inputs; col++) {
            X_val[row - num_train][col] = data[index][col];
        }
        for (int col = 0; col < num_outputs; col++) {
            Y_val[row - num_train][col] = data[index][col + num_inputs];
        }
    }

    // Free the memory allocated for the row indices array
    free(datarow_indices);
}

// Initialize the weights, biases, and activations arrays
void InitializeArrays(int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                        int num_train, double initial_range,
                        double ***W, double **b, double ***a)
{
    // Initialize weights. Need the weights for each neuron's of each layer's connections to the previous layer's neurons --> 3D array
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer]; // Neurons in the last layer is just the number of labels
        int num_neurons_previous_layer = (layer == 0) ? num_inputs : num_neurons[layer - 1]; // Neurons in the first layer is just the number of features
        W[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            W[layer][neuron] = malloc(num_neurons_previous_layer * sizeof(double));
            for (int prev_neuron = 0; prev_neuron < num_neurons_previous_layer; prev_neuron++) {
                W[layer][neuron][prev_neuron] = RandomDouble(-initial_range, initial_range); // Initialize each weight to rand between -initial_range and initial_range
            }
        }
    }

    // Initialize biases. Need the biases for each neuron in each layer
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        b[layer] = malloc(num_neurons_current_layer * sizeof(double));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            b[layer][neuron] = RandomDouble(-initial_range, initial_range);
        }
    }

    // Initialize activations. Need the activations for each neuron in each layer
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        a[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            a[layer][neuron] = malloc(num_train * sizeof(double));
        }
    }
}

// Set values for the cost and accuracy metrics for both the training and validation datasets
void CalculateMetrics(int num_train, int num_val, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons,
                      double **X_train, double **Y_train, double **X_val, double **Y_val,
                      double ***W, double **b,
                      double *accuracy_train, double *accuracy_val, double *cost_train, double *cost_val)
{

    // Declare array to store the training network activations (last layer is the model's predictions on the training dataset)
    double ***a_train = malloc((num_hidden_layers + 1) * sizeof(double **));
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        a_train[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            a_train[layer][neuron] = malloc(num_train * sizeof(double));
        }
    }
    
    // Run the model on the training dataset
    ForwardPass(num_train, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                X_train,
                W, b, a_train);

    // Allocate memory for output neurons
    double **output_neurons_train = malloc(num_train * sizeof(double *));
    for (int i = 0; i < num_train; i++) {
        output_neurons_train[i] = malloc(num_outputs * sizeof(double));
    }

    // Transpose a_train[num_hidden_layers] to output_neurons_train because this is the format that the labels are in
    for (int i = 0; i < num_outputs; i++) {
        for (int j = 0; j < num_train; j++) {
            output_neurons_train[j][i] = a_train[num_hidden_layers][i][j];
        }
    }

    // Calculate cost (MSE but calculated wrong??)
    double sum_squared_diff = 0.0;
    for (int i = 0; i < num_train; i++) {
        for (int j = 0; j < num_outputs; j++) {
            double diff = output_neurons_train[i][j] - Y_train[i][j];
            sum_squared_diff += diff * diff;
        }
    }
    *cost_train = sum_squared_diff / num_train;

    // Calculate dual binary accuracy
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

    // Declare array to store the validation network activations (last layer is the model's predictions on the validation dataset)
    double ***a_val = malloc((num_hidden_layers + 1) * sizeof(double **));
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        a_val[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            a_val[layer][neuron] = malloc(num_val * sizeof(double));
        }
    }

    // Run the model on the validation dataset
    ForwardPass(num_val, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                X_val,
                W, b, a_val);

    // Allocate memory for output neurons
    double **output_neurons_val = malloc(num_val * sizeof(double *));
    for (int i = 0; i < num_val; i++) {
        output_neurons_val[i] = malloc(num_outputs * sizeof(double));
    }

    // Transpose a_val[num_hidden_layers] to output_neurons_val because this is the format that the labels are in
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

    // Calculate dual binary accuracy
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

// Run the model on the training dataset and update the weights and biases
void ForwardPass(int num_train, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons,
                 double **X_train, double ***W, double **b, double ***a)
{
    // Calculate the activations for each neuron in each layer
    for (int layer = 0; layer < num_hidden_layers + 1; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer]; // Neurons in the last layer is just the number of labels
        int num_neurons_previous_layer = (layer == 0) ? num_inputs : num_neurons[layer - 1]; // Neurons in the first layer is just the number of features
        
        // Loop throuh each neuron in layer and calculate its activation
        for (int i = 0; i < num_neurons_current_layer; i++) {
            for (int j = 0; j < num_train; j++) {
                double sum = 0;
                for (int k = 0; k < num_neurons_previous_layer; k++) {
                    sum += W[layer][i][k] * ((layer == 0) ? X_train[j][k] : a[layer - 1][k][j]);
                }
                a[layer][i][j] = (layer == num_hidden_layers) ? Sigmoid(sum + b[layer][i]) : Sigmoid(sum + b[layer][i]); // Activation function
            }
        }
    }
}

// Update the weights and biases using the learning rate and the partial derivatives of the cost function (gradient)
void BackwardPass(double learning_rate, int num_train, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons,
                  double **X_train, double **Y_train, double ***W, double **b, double ***a)
{
    // Array to store the gradient of the cost function wrt activations
    double ***PL = malloc((num_hidden_layers + 1) * sizeof(double **));
    for (int layer = 0; layer < num_hidden_layers + 1; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        PL[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            PL[layer][neuron] = malloc(num_train * sizeof(double));
        }
    }

    // Compute PL for the output layer by taking gradient of cost
    for (int i = 0; i < num_outputs; i++) {
        for (int j = 0; j < num_train; j++) {
            PL[num_hidden_layers][i][j] = (a[num_hidden_layers][i][j] - Y_train[j][i]) * (1 - a[num_hidden_layers][i][j]) * a[num_hidden_layers][i][j];
        }
    }

    // Compute PL for the each hidden layer
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
                W[layer][i][j] -= learning_rate * sum;
            }
        }

        for (int i = 0; i < num_neurons_current_layer; i++) {
            double sum = 0.0;
            for (int j = 0; j < num_train; j++) {
                sum += PL[layer][i][j];
            }
            b[layer][i] -= learning_rate * sum;
        }
    }

    // Free memory for PL
    for (int layer = 0; layer < num_hidden_layers + 1; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            free(PL[layer][neuron]);
        }
        free(PL[layer]);
    }
    free(PL);
}

// Train model and evaluate performance
void Evaluation(int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, char *filename,
                int epochs, double learning_rate, double initial_range, int num_train, int num_val, double train_split,
                double **X_train, double **Y_train, double **X_val, double **Y_val)
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

    // Ask user if they want to download ANN
    DownloadANN(epochs, learning_rate, initial_range, filename, train_split, num_train, num_val, 
                 num_inputs, num_outputs, num_hidden_layers, num_neurons, 
                 X_train, Y_train, X_val, Y_val, W, b);

    // Free the dynamically allocated memory for activations
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            free(a[layer][neuron]);
        }
        free(a[layer]);
    }
    free(a);

    // Free the dynamically allocated memory for weights and biases
    for (int layer = 0; layer < num_hidden_layers; layer++) {
        free(W[layer]);
        free(b[layer]);
    }
    free(W);
    free(b);
}

// Download the ANN to a txt file if the user wants
void DownloadANN(int epochs, double learning_rate, double initial_range, char *filename, double train_split, int num_train,
                  int num_val, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                  double **X_train, double **Y_train, double **X_val, double **Y_val, double ***W, double **b) 
{
    char userResponse[3];
    int validResponse = 0;

    // Continue looping until user enters a valid answer
    while (!validResponse) {
        printf("Do you want to download the ANN? (yes/no): ");
        scanf("%s", userResponse);

        // These are the valid responses, otherwise we ask again
        if (strcmp(userResponse, "yes") == 0 || strcmp(userResponse, "no") == 0 || strcmp(userResponse, "y") == 0 || strcmp(userResponse, "n") == 0 || strcmp(userResponse, "Y") == 0 || strcmp(userResponse, "N") == 0 || strcmp(userResponse, "Yes") == 0 || strcmp(userResponse, "No") == 0) {
            validResponse = 1;
        } else {
            printf("Sorry, '%s' is an invalid response. Please enter 'yes' or 'no'.\n", userResponse);
        }
    }

    if (strcmp(userResponse, "yes") == 0 || strcmp(userResponse, "y") == 0 || strcmp(userResponse, "Y") == 0 || strcmp(userResponse, "Yes") == 0) {
        // Create folder if it doesn't already exist
        mkdir("Downloaded Data", 0777);

        // Open file inside the folder to write data into
        FILE *file = fopen("Downloaded Data/ANN_data.txt", "w");
        if (file == NULL) {
            printf("Error opening file!\n");
            return;
        }

        // Write the ANN's information to the file
        fprintf(file, "-- NEURAL NETWORK ARCHITECTURE --\n");
        fprintf(file, "Epochs: %d\n", epochs);
        fprintf(file, "Learning Rate: %f\n", learning_rate);
        fprintf(file, "Weight and Bias Initialization Range: (%f, %f)\n", -initial_range, initial_range);
        fprintf(file, "Dataset: %s\n", filename);
        fprintf(file, "Train Split (Proportion): %f\n", train_split);
        fprintf(file, "Training Dataset Samples: %d\n", num_train);
        fprintf(file, "Validation Dataset Samples: %d\n", num_val);
        fprintf(file, "Input Neurons (Features): %d\n", num_inputs);
        fprintf(file, "Output Neurons (Labels): %d\n", num_outputs);
        fprintf(file, "Hidden Layers: %d\n", num_hidden_layers);
        fprintf(file, "Quantity of Neurons in Sequential Hidden Layers: ");
        for (int i = 0; i < num_hidden_layers; i++) {
            fprintf(file, "%d ", num_neurons[i]);
        } 

        // Compute metrics so that we can write them in
        double cost_train, cost_val, accuracy_train, accuracy_val;
        CalculateMetrics(num_train, num_val, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                        X_train, Y_train, X_val, Y_val,
                        W, b,
                        &accuracy_train, &accuracy_val, &cost_train, &cost_val);

        // Write the model performance metrics to the file
        fprintf(file, "\n\n-- MODEL PERFORMANCE --\n");
        fprintf(file, "Train Cost:      %lf     Accuracy: %.2f%%\n", cost_train, accuracy_train * 100);
        fprintf(file, "Validation Cost: %lf     Accuracy: %.2f%%\n", cost_val, accuracy_val * 100);

        // Write the weights to the file
        fprintf(file, "\n-- WEIGHTS --\n");
        for (int layer = 0; layer <= num_hidden_layers; layer++) {
            int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
            int num_neurons_previous_layer = (layer == 0) ? num_inputs : num_neurons[layer - 1];
            fprintf(file, "Layer %d:\n", layer + 1);
            for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
                fprintf(file, "Neuron %d: ", neuron + 1);
                for (int prev_neuron = 0; prev_neuron < num_neurons_previous_layer; prev_neuron++) {
                    fprintf(file, "%lf ", W[layer][neuron][prev_neuron]);
                }
                fprintf(file, "\n");
            }
            fprintf(file, "\n");
        }

        // Write the biases to the file
        fprintf(file, "\n-- BIASES --\n");
        for (int layer = 0; layer <= num_hidden_layers; layer++) {
            int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
            fprintf(file, "Layer %d: ", layer + 1);
            for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
                fprintf(file, "%lf ", b[layer][neuron]);
            }
            fprintf(file, "\n");
        }

        // Close the file
        fclose(file);

        printf("The ANN has been downloaded to the following directory: Downloaded Data/ANN_Data.txt.\n");
    }
}