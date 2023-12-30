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
InputData ReadFile(int num_cols) {
    FILE *file;
    int valid_response = 0;
    char filename[100];
    while (valid_response == 0) {
        printf("Please enter the file name containing the dataset: ");

        // Get the filename from the user
        if (scanf("%99s", filename) != 1) {
            printf("Error reading input.\n");
            continue;
        }

        // Check if the file name ends with .txt
        char *extension = strrchr(filename, '.');
        if (extension == NULL || strcmp(extension, ".txt") != 0) {
            printf("Invalid file name. Please enter a file name ending with .txt.\n");
            continue;
        }

        // Open the file and read
        file = fopen(filename, "r");
        if (file == NULL) {
            printf("Sorry, I could not open '%s'. Please ensure that it is in the proper directory.\n", filename);
        } else {
            valid_response = 1;
        }
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
    double **data = (double **)malloc(num_rows * sizeof(double *));
    for (int row = 0; row < num_rows; row++) {
        data[row] = (double *)malloc(num_cols * sizeof(double));
    }

    // Read the file again from beginning
    fseek(file, 0, SEEK_SET);

    // Transcribe the data from the file into the data array
    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols; col++) {
            if (fscanf(file, "%lf", &data[row][col]) != 1) {
                fprintf(stderr, "Error reading input from file.\n");
                exit(1);
            }
        }
    }

    // Close the file
    fclose(file);
    printf("\n");

    // Create and return DataResult struct
    InputData input_data;
    input_data.data = data;
    input_data.num_rows = num_rows;
    
    input_data.filename = malloc(strlen(filename) + 1);
    strcpy(input_data.filename, filename);

    return input_data;
}

// Split the data into training and validation sets
void OrganizeData(int num_train, int num_inputs, int num_outputs, int num_rows,
                  double **data, double **X_train, double **Y_train, double **X_val, double **Y_val)
{
    // Array to store indices of each row of data
    int *datarow_indices = malloc(num_rows * sizeof(int));
    for (int row = 0; row < num_rows; row++) {
        datarow_indices[row] = row;
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
                      double **X_train, double **Y_train, double **X_val, double **Y_val, double ***W, double **b,
                      double *accuracy_train, double *accuracy_val, double *mse_train, double *mse_val, double *log_loss_train, double *log_loss_val, double *r2_train, double *r2_val)
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
                X_train, W, b, a_train);

    // Allocate memory for output neurons
    double **output_neurons_train = malloc(num_train * sizeof(double *));
    for (int training_example = 0; training_example < num_train; training_example++) {
        output_neurons_train[training_example] = malloc(num_outputs * sizeof(double));
    }

    // Transpose a_train[num_hidden_layers] to output_neurons_train because this is the format that the labels are in
    for (int i = 0; i < num_outputs; i++) {
        for (int j = 0; j < num_train; j++) {
            output_neurons_train[j][i] = a_train[num_hidden_layers][i][j];
        }
    }

    // Calculate MSE
    double sum_squared_diff = 0.0;
    for (int training_example = 0; training_example < num_train; training_example++) {
        for (int output_neuron = 0; output_neuron < num_outputs; output_neuron++) {
            double diff = output_neurons_train[training_example][output_neuron] - Y_train[training_example][output_neuron];
            sum_squared_diff += diff * diff;
        }
    }
    *mse_train = sum_squared_diff / (num_train * num_outputs);

    // Calculate dual binary accuracy
    int correct_predictions = 0;
    for (int training_example = 0; training_example < num_train; training_example++) {
        int all_correct = 1;
        for (int output_neuron = 0; output_neuron < num_outputs; output_neuron++) {
            if ((output_neurons_train[training_example][output_neuron] >= 0.5 && Y_train[training_example][output_neuron] == 0) || (output_neurons_train[training_example][output_neuron] < 0.5 && Y_train[training_example][output_neuron] == 1)) {
                all_correct = 0;
                break;
            }
        }
        correct_predictions += all_correct;
    }
    *accuracy_train = (double)correct_predictions / num_train;

    // Calculate log loss
    *log_loss_train = 0.0;
    for (int training_example = 0; training_example < num_train; training_example++) {
        for (int output_neuron = 0; output_neuron < num_outputs; output_neuron++) {
            double y = Y_train[training_example][output_neuron];
            double a = output_neurons_train[training_example][output_neuron];
            *log_loss_train += y * log(a + 1e-16) + (1 - y) * log(1 - a + 1e-16);
        }
    }
    *log_loss_train /= -num_train;

    // Calculate R2
    double sum_y = 0.0;
    double sum_yhat = 0.0;
    double sum_squared_y = 0.0;
    double sum_squared_yhat = 0.0;
    double sum_y_yhat = 0.0;
    for (int training_example = 0; training_example < num_train; training_example++) {
        for (int output_neuron = 0; output_neuron < num_outputs; output_neuron++) {
            sum_y += Y_train[training_example][output_neuron];
            sum_yhat += output_neurons_train[training_example][output_neuron];
            sum_squared_y += Y_train[training_example][output_neuron] * Y_train[training_example][output_neuron];
            sum_squared_yhat += output_neurons_train[training_example][output_neuron] * output_neurons_train[training_example][output_neuron];
            sum_y_yhat += Y_train[training_example][output_neuron] * output_neurons_train[training_example][output_neuron];
        }
    }
    double mean_y = sum_y / (num_train * num_outputs);
    double mean_yhat = sum_yhat / (num_train * num_outputs);
    double numerator = sum_y_yhat - num_train * num_outputs * mean_y * mean_yhat;
    double denominator = sqrt((sum_squared_y - num_train * num_outputs * mean_y * mean_y) * (sum_squared_yhat - num_train * num_outputs * mean_yhat * mean_yhat));
    *r2_train = numerator / denominator;

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
                X_val, W, b, a_val);

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

    // Calculate MSE
    sum_squared_diff = 0.0;
    for (int i = 0; i < num_val; i++) {
        for (int j = 0; j < num_outputs; j++) {
            double diff = output_neurons_val[i][j] - Y_val[i][j];
            sum_squared_diff += diff * diff;
        }
    }
    *mse_val = sum_squared_diff / (num_val * num_outputs);

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

    // Calculate log loss
    *log_loss_val = 0.0;
    for (int i = 0; i < num_val; i++) {
        for (int j = 0; j < num_outputs; j++) {
            double y = Y_val[i][j];
            double a = output_neurons_val[i][j];
            *log_loss_val += y * log(a + 1e-16) + (1 - y) * log(1 - a + 1e-16);
        }
    }
    *log_loss_val /= -num_val;

    // Calculate R2
    sum_y = 0.0;
    sum_yhat = 0.0;
    sum_squared_y = 0.0;
    sum_squared_yhat = 0.0;
    sum_y_yhat = 0.0;
    for (int i = 0; i < num_val; i++) {
        for (int j = 0; j < num_outputs; j++) {
            sum_y += Y_val[i][j];
            sum_yhat += output_neurons_val[i][j];
            sum_squared_y += Y_val[i][j] * Y_val[i][j];
            sum_squared_yhat += output_neurons_val[i][j] * output_neurons_val[i][j];
            sum_y_yhat += Y_val[i][j] * output_neurons_val[i][j];
        }
    }
    mean_y = sum_y / (num_val * num_outputs);
    mean_yhat = sum_yhat / (num_val * num_outputs);
    numerator = sum_y_yhat - num_val * num_outputs * mean_y * mean_yhat;
    denominator = sqrt((sum_squared_y - num_val * num_outputs * mean_y * mean_y) * (sum_squared_yhat - num_val * num_outputs * mean_yhat * mean_yhat));
    *r2_val = numerator / denominator;

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
    for (int validation_example = 0; validation_example < num_val; validation_example++) {
        free(output_neurons_val[validation_example]);
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
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            for (int training_example = 0; training_example < num_train; training_example++) {
                double sum = 0;
                for (int prev_neuron = 0; prev_neuron < num_neurons_previous_layer; prev_neuron++) {
                    sum += W[layer][neuron][prev_neuron] * ((layer == 0) ? X_train[training_example][prev_neuron] : a[layer - 1][prev_neuron][training_example]);
                }
                a[layer][neuron][training_example] = (layer == num_hidden_layers) ? Sigmoid(sum + b[layer][neuron]) : tanh(sum + b[layer][neuron]); // Activation function
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
    for (int num_output = 0; num_output < num_outputs; num_output++) {
        for (int training_example = 0; training_example < num_train; training_example++) {
            PL[num_hidden_layers][num_output][training_example] = (a[num_hidden_layers][num_output][training_example] - Y_train[training_example][num_output]) * (1 - a[num_hidden_layers][num_output][training_example]) * a[num_hidden_layers][num_output][training_example];
        }
    }

    // Compute PL for the each hidden layer
    for (int layer = num_hidden_layers - 1; layer >= 0; layer--) {
        int num_neurons_current_layer = num_neurons[layer];
        int num_neurons_next_layer = (layer == num_hidden_layers - 1) ? num_outputs : num_neurons[layer + 1];

        double *a_squared_complement = malloc(num_neurons_current_layer * num_train * sizeof(double));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            for (int training_example = 0; training_example < num_train; training_example++) {
                a_squared_complement[neuron * num_train + training_example] = 1 - a[layer][neuron][training_example] * a[layer][neuron][training_example];
            }
        }

        double *W_PL = malloc(num_neurons_current_layer * num_train * sizeof(double));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            for (int training_example = 0; training_example < num_train; training_example++) {
                double sum = 0.0;
                for (int next_neuron = 0; next_neuron < num_neurons_next_layer; next_neuron++) {
                    sum += W[layer + 1][next_neuron][neuron] * PL[layer + 1][next_neuron][training_example];
                }
                W_PL[neuron * num_train + training_example] = sum;
            }
        }

        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            for (int training_example = 0; training_example < num_train; training_example++) {
                PL[layer][neuron][training_example] = a_squared_complement[neuron * num_train + training_example] * W_PL[neuron * num_train + training_example];
            }
        }

        free(a_squared_complement);
        free(W_PL);
    }

    // Update weights and biases using learning_rate and PL
    for (int layer = 0; layer < num_hidden_layers + 1; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        int num_neurons_previous_layer = (layer == 0) ? num_inputs : num_neurons[layer - 1];

        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            for (int prev_neuron = 0; prev_neuron < num_neurons_previous_layer; prev_neuron++) {
                double sum = 0.0;
                for (int k = 0; k < num_train; k++) {
                    sum += PL[layer][neuron][k] * ((layer == 0) ? X_train[k][prev_neuron] : a[layer - 1][prev_neuron][k]);
                }
                W[layer][neuron][prev_neuron] -= learning_rate * sum;
            }
        }

        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            double sum = 0.0;
            for (int training_example = 0; training_example < num_train; training_example++) {
                sum += PL[layer][neuron][training_example];
            }
            b[layer][neuron] -= learning_rate * sum;
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

    // Write metrics to file
    mkdir("Exported Data", 0777);
    FILE *file = fopen("Data Visualizations/epoch_metrics.csv", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fprintf(file, "Epoch,Train MSE,Validation MSE,Train Accuracy,Validation Accuracy,Train Log Loss,Validation Log Loss,Train R²,Validation R²\n");
    fclose(file);

    // Initialize arrays
    InitializeArrays(num_inputs, num_outputs, num_hidden_layers, num_neurons, num_train, initial_range, W, b, a);

    for (int ep = 1; ep <= epochs; ep++) {

        ForwardPass(num_train, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                    X_train, W, b, a);
        
        BackwardPass(learning_rate, num_train, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                    X_train, Y_train, W, b, a);

        // Evaluate ANN every 100 epochs
        if (ep % 100 == 0) {
            double mse_train, mse_val, accuracy_train, accuracy_val, log_loss_train, log_loss_val, r2_train, r2_val;

            CalculateMetrics(num_train, num_val, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                            X_train, Y_train, X_val, Y_val, W, b,
                            &accuracy_train, &accuracy_val, &mse_train, &mse_val, &log_loss_train, &log_loss_val, &r2_train, &r2_val);

            printf("Epoch %d:\n", ep);
            printf("Train Metrics      || MSE: %lf || Accuracy: %.2f%% || Log Loss: %lf || R²: %lf\n", mse_train, accuracy_train * 100, log_loss_train, r2_train);
            printf("Validation Metrics || MSE: %lf || Accuracy: %.2f%% || Log Loss: %lf || R²: %lf\n", mse_val, accuracy_val * 100, log_loss_val, r2_val);
            printf("\n");

            // Write metrics to file
            FILE *file = fopen("Data Visualizations/epoch_metrics.csv", "a");
            if (file == NULL) {
                printf("Error opening file!\n");
                return;
            }
            fprintf(file, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", ep, mse_train, mse_val, accuracy_train, accuracy_val, log_loss_train, log_loss_val, r2_train, r2_val);
            fclose(file);

        }
    }

    // Generate coordinates in CSV for ROC curve
    ROC(num_train, num_val, num_inputs, num_outputs, num_hidden_layers, num_neurons,
        Y_train, X_val, Y_val, W, b, a);

    // Ask user if they want to download ANN
    DownloadANN(epochs, learning_rate, initial_range, filename, train_split, num_train, num_val, 
                 num_inputs, num_outputs, num_hidden_layers, num_neurons, 
                 X_train, Y_train, X_val, Y_val, W, b);

    // Ask user if they want to make predictions on a new input data file
    MakePredictions(W, b, num_inputs, num_hidden_layers, num_neurons, num_outputs);

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

// Calculate the true positive rate and false positive rate for the ROC curve
void ROC(int num_train, int num_val, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons,
         double **Y_train, double **X_val, double **Y_val, double ***W, double **b, double ***a)
{
    // Create folder if it doesn't already exist
    mkdir("Data Visualizations", 0777);

    // Open file inside the folder to write data into
    FILE *file_train = fopen("Data Visualizations/ROC_train.csv", "w");
    if (file_train == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fprintf(file_train, "Threshold,False Positive Rate (x),True Positive Rate (y)\n");

    // Create array for the ANN's predictions in one dimension
    Prediction *predictions_train = malloc(num_train * num_outputs * sizeof(Prediction));
    for (int i = 0; i < num_train; i++) {
        for (int j = 0; j < num_outputs; j++) {
            predictions_train[i*num_outputs + j].true_label = Y_train[i][j];
            predictions_train[i*num_outputs + j].prediction = a[num_hidden_layers][j][i];
        }
    }

    // Iterate through threshold to generate ROC coordinates
    for (int threshold = 0; threshold <= 1000; threshold++) {
        int true_positives = 0;
        int false_positives = 0;
        int true_negatives = 0;
        int false_negatives = 0;
        double true_positive_rate, false_positive_rate;

        // Calculate the number of true positives, false positives, true negatives, and false negatives
        for (int i = 0; i < num_train * num_outputs; i++) {
            if (predictions_train[i].true_label == 1 && predictions_train[i].prediction >= (double)threshold/1000) {
                true_positives++;
            } else if (predictions_train[i].true_label == 1 && predictions_train[i].prediction < (double)threshold/1000) {
                false_negatives++;
            } else if (predictions_train[i].true_label == 0 && predictions_train[i].prediction >= (double)threshold/1000) {
                false_positives++;
            } else if (predictions_train[i].true_label == 0 && predictions_train[i].prediction < (double)threshold/1000) {
                true_negatives++;
            }
        }
        // Calculate TPR and FPR for ROC coordinates
        true_positive_rate = (double)true_positives / (true_positives + false_negatives);
        false_positive_rate = (double)false_positives / (false_positives + true_negatives);
        // printf("%f || (%f, %f)\n", (double)threshold/1000, false_positive_rate, true_positive_rate);
        fprintf(file_train, "%f,%f,%f\n", (double)threshold/1000, false_positive_rate, true_positive_rate);
    }

    fclose(file_train);
    free(predictions_train);

    // Run the model on the validation dataset
    double ***a_val = malloc((num_hidden_layers + 1) * sizeof(double **));
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        a_val[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            a_val[layer][neuron] = malloc(num_val * sizeof(double));
        }
    }
    ForwardPass(num_val, num_inputs, num_outputs, num_hidden_layers, num_neurons, X_val, W, b, a_val);

    // Open file inside the folder to write data into
    FILE *file_val = fopen("Data Visualizations/ROC_validation.csv", "w");
    if (file_val == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fprintf(file_val, "Threshold,False Positive Rate (x),True Positive Rate (y)\n");

    // Create array for the ANN's predictions in one dimension
    Prediction *predictions_val = malloc(num_val * num_outputs * sizeof(Prediction));
    for (int i = 0; i < num_val; i++) {
        for (int j = 0; j < num_outputs; j++) {
            predictions_val[i*num_outputs + j].true_label = Y_val[i][j];
            predictions_val[i*num_outputs + j].prediction = a_val[num_hidden_layers][j][i];
        }
    }

    // Iterate through threshold to generate ROC coordinates
    for (int threshold = 0; threshold <= 1000; threshold++) {
        int true_positives = 0;
        int false_positives = 0;
        int true_negatives = 0;
        int false_negatives = 0;
        double true_positive_rate, false_positive_rate;

        // Calculate the number of true positives, false positives, true negatives, and false negatives
        for (int i = 0; i < num_val * num_outputs; i++) {
            if (predictions_val[i].true_label == 1 && predictions_val[i].prediction >= (double)threshold/1000) {
                true_positives++;
            } else if (predictions_val[i].true_label == 1 && predictions_val[i].prediction < (double)threshold/1000) {
                false_negatives++;
            } else if (predictions_val[i].true_label == 0 && predictions_val[i].prediction >= (double)threshold/1000) {
                false_positives++;
            } else if (predictions_val[i].true_label == 0 && predictions_val[i].prediction < (double)threshold/1000) {
                true_negatives++;
            }
        }
        // Calculate TPR and FPR for ROC coordinates
        true_positive_rate = (double)true_positives / (true_positives + false_negatives);
        false_positive_rate = (double)false_positives / (false_positives + true_negatives);
        // printf("%f || (%f, %f)\n", (double)threshold/1000, false_positive_rate, true_positive_rate);
        fprintf(file_val, "%f,%f,%f\n", (double)threshold/1000, false_positive_rate, true_positive_rate);
    }

    fclose(file_val);
    free(predictions_val);
    for (int layer = 0; layer <= num_hidden_layers; layer++) {
        int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
            free(a_val[layer][neuron]);
        }
        free(a_val[layer]);
    }
    free(a_val);
}

// Download the ANN to a txt file if the user wants
void DownloadANN(int epochs, double learning_rate, double initial_range, char *filename, double train_split, int num_train,
                  int num_val, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                  double **X_train, double **Y_train, double **X_val, double **Y_val, double ***W, double **b) 
{
    char userResponse[100];
    int valid_response = 0;

    // Continue looping until user enters a valid answer
    while (!valid_response) {
        printf("Do you want to download the ANN? (yes/no): ");
        if (scanf("%99s", userResponse) != 1) {
            fprintf(stderr, "Error reading input.\n");
            exit(1);
        }

        // These are the valid responses, otherwise we ask again
        if (strcmp(userResponse, "yes") == 0 || strcmp(userResponse, "no") == 0 || strcmp(userResponse, "y") == 0 || strcmp(userResponse, "n") == 0 || strcmp(userResponse, "Y") == 0 || strcmp(userResponse, "N") == 0 || strcmp(userResponse, "Yes") == 0 || strcmp(userResponse, "No") == 0) {
            valid_response = 1;
        } else {
            printf("Sorry, '%s' is an invalid response. Please enter 'yes' or 'no'.\n", userResponse);
        }
    }

    if (strcmp(userResponse, "yes") == 0 || strcmp(userResponse, "y") == 0 || strcmp(userResponse, "Y") == 0 || strcmp(userResponse, "Yes") == 0) {
        // Create folder if it doesn't already exist
        mkdir("Exported Data", 0777);

        // Open file inside the folder to write data into
        FILE *file = fopen("Exported Data/ANN_data.txt", "w");
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
        fprintf(file, "Activation function: Sigmoid\n");
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
        double mse_train, mse_val, accuracy_train, accuracy_val, log_loss_train, log_loss_val, r2_train, r2_val;
        CalculateMetrics(num_train, num_val, num_inputs, num_outputs, num_hidden_layers, num_neurons,
                        X_train, Y_train, X_val, Y_val, W, b,
                        &accuracy_train, &accuracy_val, &mse_train, &mse_val, &log_loss_train, &log_loss_val, &r2_train, &r2_val);

        // Write the model performance metrics to the file
        fprintf(file, "\n\n-- MODEL PERFORMANCE --\n");
        fprintf(file, "Train Metrics      || MSE: %lf || Accuracy: %.2f%% || Log Loss: %lf || R²: %lf\n", mse_train, accuracy_train * 100, log_loss_train, r2_train);
        fprintf(file, "Validation Metrics || MSE: %lf || Accuracy: %.2f%% || Log Loss: %lf || R²: %lf\n", mse_val, accuracy_val * 100, log_loss_val, r2_val);

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

        printf("The ANN has been downloaded to the following directory: Exported Data/ANN_Data.txt.\n");
    }
}

// Make predictions on a given file
void MakePredictions(double ***W, double **b, int num_inputs, int num_hidden_layers, int *num_neurons, int num_outputs) {
    char userResponse[100];
    int valid_response = 0;

    // Continue looping until user enters a valid answer
    while (!valid_response) {
        printf("Do you want to use this ANN make predictions on another file of data? (yes/no): ");
        if (scanf("%99s", userResponse) != 1) {
            fprintf(stderr, "Error reading input.\n");
            exit(1);
        }

        // These are the valid responses, otherwise we ask again
        if (strcmp(userResponse, "yes") == 0 || strcmp(userResponse, "no") == 0 || strcmp(userResponse, "y") == 0 || strcmp(userResponse, "n") == 0 || strcmp(userResponse, "Y") == 0 || strcmp(userResponse, "N") == 0 || strcmp(userResponse, "Yes") == 0 || strcmp(userResponse, "No") == 0) {
            valid_response = 1;
        } else {
            printf("Sorry, '%s' is an invalid response. Please enter 'yes' or 'no'.\n", userResponse);
        }
    }

    if (strcmp(userResponse, "yes") == 0 || strcmp(userResponse, "y") == 0 || strcmp(userResponse, "Y") == 0 || strcmp(userResponse, "Yes") == 0) {
        // Continue looping until user enters a valid answer
        FILE *input_file = NULL;
        valid_response = 0;
        while (!valid_response) {
            printf("Enter the name of the file that you would like to make predictions on: ");
            if (scanf("%99s", userResponse) != 1) {
                fprintf(stderr, "Error reading input.\n");
                exit(1);
            }

            input_file = fopen(userResponse, "r");
            if (input_file == NULL) {
                printf("Sorry, I could not open '%s'. Please ensure it is in the proper directory!\n", userResponse);
            } else {
                valid_response = 1;
            }
        }

        // Count the number of rows in input_file
        int num_rows = 0;
        char character;
        while ((character = fgetc(input_file)) != EOF) {
            if (character == '\n') {
                num_rows++;
            }
        }

        // Create folder if it doesn't already exist
        mkdir("Exported Data", 0777);

        // Remove the extension from the input file name
        char *filename_without_ext = strdup(userResponse); // Duplicate the string
        char *dot = strrchr(filename_without_ext, '.');
        if (dot) *dot = '\0'; // If there was an extension, remove it
        char output_filename[100];
        sprintf(output_filename, "%s_predictions.csv", filename_without_ext);
        char output_filepath[100];
        sprintf(output_filepath, "Exported Data/%s", output_filename);
        free(filename_without_ext);

        FILE *output_file = fopen(output_filepath, "w");
        if (output_file == NULL) {
            printf("Error opening output file!\n");
            return;
        }

        for (int i = 0; i < num_inputs; i++) {
            fprintf(output_file, "Input Variable %d,", i+1);
        }
        for (int i = 0; i < num_outputs; i++) {
            fprintf(output_file, "Correct Output %d,", i+1);
        }
        for (int i = 0; i < num_outputs; i++) {
            fprintf(output_file, "Predicted Output %d,", i+1);
        }
        fseek(output_file, -1, SEEK_CUR);
        fprintf(output_file, "\n");

        // Allocate memory for 2D array and fill with file data
        double **data = (double **)malloc(num_rows * sizeof(double *));
        for (int row = 0; row < num_rows; row++) {
            data[row] = (double *)malloc((num_inputs + num_outputs) * sizeof(double));
        }

        rewind(input_file);

        // Transcribe the data from the file into the data array
        for (int row = 0; row < num_rows; row++) {
            for (int col = 0; col < (num_inputs + num_outputs); col++) {
                if (fscanf(input_file, "%lf", &data[row][col]) != 1) {
                    fprintf(stderr, "Error reading input from file '%s'.\n", userResponse);
                    exit(1);
                }
            }
        }

        // Close the input file
        fclose(input_file);

        // Allocate memory for a
        double ***a = malloc((num_hidden_layers + 1) * sizeof(double **));
        for (int layer = 0; layer < num_hidden_layers + 1; layer++) {
            int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
            a[layer] = malloc(num_neurons_current_layer * sizeof(double *));
            for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
                a[layer][neuron] = malloc(num_rows * sizeof(double));
            }
        }

        // Make predictions based on these inputs
        ForwardPass(num_rows, num_inputs, num_outputs, num_hidden_layers, num_neurons, data, W, b, a);
        
        // Print the data array and make predictions
        for (int row = 0; row < num_rows; row++) {
            for (int col = 0; col < (num_inputs + num_outputs); col++) {
                fprintf(output_file, "%lf,", data[row][col]);
            }

            // Add the predictions to the file
            for (int output = 0; output < num_outputs; output++) {
                fprintf(output_file, "%lf,", a[num_hidden_layers][output][row]);
            }
            fseek(output_file, -1, SEEK_CUR);
            fprintf(output_file, "\n");
        }

        // Close the output file
        fclose(output_file);

        printf("Predictions have been added to the file: %s\n", output_filepath);

        // Free memory for data
        for (int row = 0; row < num_rows; row++) {
            free(data[row]);
        }
        free(data);

        // Free memory for a
        for (int layer = 0; layer <= num_hidden_layers; layer++) {
            int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
            for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
                free(a[layer][neuron]);
            }
            free(a[layer]);
        }
        free(a);
    }
}