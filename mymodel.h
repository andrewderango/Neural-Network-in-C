#ifndef MYMODEL_H
#define MYMODEL_H

double sigmoid(double x);

double random_double(double min, double max);

void ReadFile(int MAX_ROWS, int MAX_COLS, double data[MAX_ROWS][MAX_COLS], char* filename);

void ForwardPass(int num_train, int num_inputs, int num_outputs, int num_neurons_layer2, int num_neurons_layer3, 
                double X_train[][num_inputs], double Y_train[][num_outputs],
                double W2[][num_inputs], double W3[][num_neurons_layer2], double W4[][num_neurons_layer3],
                double b2[][1], double b3[][1], double b4[][1],
                double a2[][num_train], double a3[][num_train], double a4[][num_train]);

void BackwardPass(double Learning_rate, int num_train, int num_inputs, int num_outputs, int num_neurons_layer2, int num_neurons_layer3, 
                double X_train[][num_inputs], double Y_train[][num_outputs],
                double W2[][num_inputs], double W3[][num_neurons_layer2], double W4[][num_neurons_layer3],
                double b2[][1], double b3[][1], double b4[][1],
                double a2[][num_train], double a3[][num_train], double a4[][num_train]);

void Evaluation(int num_inputs, int num_outputs, int num_neurons_layer2, int num_neurons_layer3, 
                int epochs, double learning_rate, double initial_range, int num_train, int num_val,
                double X_train[num_train][num_inputs], double Y_train[num_train][num_outputs], 
                double X_val[num_val][num_inputs], double Y_val[num_val][num_outputs]);

#endif