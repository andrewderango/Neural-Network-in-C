#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sodium.h> // For randombytes_buf function
#include "mymodel.h"

// #define num_inputs 2          // N1 = 2
// #define num_neurons_layer2 40 // N2 = 40
// #define num_neurons_layer3 20 // N3 = 20
// #define num_outputs 2         // N4 = 2
#define initial_range 0.2

// #define Learning_rate 0.005 // 0 - 1, 0.01 - 0.0001
// #define epochs 100000

#define MAX_ROWS 48120
#define MAX_COLS 4
// #define train_split 0.003 // 0.3 percent of data will be used for train

// QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ
//  The same as ForwardPass, do the function BackwardPass() here

int main()
{
    // Initialize the 2D array to store the data
    double data[MAX_ROWS][MAX_COLS];
    char *filename = "data.txt";

    // There should be a better way to handle these define elements
    int max_rows = MAX_ROWS;
    int max_cols = MAX_COLS;
    double init_range = initial_range;

    // inputs
    int num_inputs = 2;
    int num_neurons_layer2 = 40;
    int num_neurons_layer3 = 20;
    int num_outputs = 2;
    int epochs = 100000;
    double learning_rate = 0.005;
    double train_split = 0.003;
    
    ReadFile(max_rows, max_cols, data, filename);

    // Should this be in ReadFile? It seems like it should but the assignment says otherwise

    int num_train = MAX_ROWS *train_split+1;
    int num_val = MAX_ROWS *(1-train_split);

    double X_train[num_train][num_inputs];
    double Y_train[num_train][num_outputs];
    double X_val[num_val][num_inputs];
    double Y_val[num_val][num_outputs];

    for (int row = 0; row < num_train; row++)
    {
        X_train[row][0] = data[row][0];
        X_train[row][1] = data[row][1];
        Y_train[row][0] = data[row][2];
        Y_train[row][1] = data[row][3];
    }

    for (int row = num_train; row < MAX_ROWS; row++)
    {
        printf("%d %d \n", row-num_train, row);
        X_val[row - num_train][0] = data[row][0];
        X_val[row - num_train][1] = data[row][1];
        Y_val[row - num_train][0] = data[row][2];
        Y_val[row - num_train][1] = data[row][3];
    }

    Evaluation(num_inputs, num_outputs, num_neurons_layer2, num_neurons_layer3, 
                epochs, learning_rate, init_range, num_train, num_val, 
                X_train, Y_train, X_val, Y_val);
}
