#ifndef MYMODEL_H
#define MYMODEL_H

void seed_sodium_library();

double sigmoid(double x);

double random_double(double min, double max);

void ReadFile(int MAX_ROWS, int MAX_COLS, double data[MAX_ROWS][MAX_COLS], char* filename);

void ForwardPass(int num_train, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                double X_train[][num_inputs], double Y_train[][num_outputs],
                double ***W, double **b, double ***a);

void BackwardPass(double Learning_rate, int num_train, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                double X_train[][num_inputs], double Y_train[][num_outputs],
                double ***W, double **b, double ***a);

void Evaluation(int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons,
                int epochs, double learning_rate, double initial_range, int num_train, int num_val,
                double X_train[num_train][num_inputs], double Y_train[num_train][num_outputs], 
                double X_val[num_val][num_inputs], double Y_val[num_val][num_outputs]);

#endif