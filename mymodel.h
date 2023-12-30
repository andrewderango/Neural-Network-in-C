#ifndef MYMODEL_H
#define MYMODEL_H

typedef struct {
    double **data;
    int num_rows;
    char *filename;
} InputData;

typedef struct {
    double prediction;
    int true_label;
} Prediction;

double Sigmoid(double x);

double RandomDouble(double min, double max);

InputData ReadFile(int num_cols);

void OrganizeData(int num_train, int num_inputs, int num_outputs, int num_rows,
                  double **data, double **X_train, double **Y_train, double **X_val, double **Y_val);

void InitializeArrays(int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                        int num_train, double initial_range,
                        double ***W, double **b, double ***a);

void CalculateMetrics(int num_train, int num_val, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons,
                      double **X_train, double **Y_train, double **X_val, double **Y_val, double ***W, double **b,
                      double *accuracy_train, double *accuracy_val, double *cost_train, double *cost_val, double *log_loss_train, double *log_loss_val, double *r2_train, double *r2_val);

void ForwardPass(int num_train, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons,
                 double **X_train, double ***W, double **b, double ***a);

void BackwardPass(double learning_rate, int num_train, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons,
                  double **X_train, double **Y_train, double ***W, double **b, double ***a);

void Evaluation(int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, char *filename,
                int epochs, double learning_rate, double initial_range, int num_train, int num_val, double train_split,
                double **X_train, double **Y_train, double **X_val, double **Y_val);

void ROC(int num_train, int num_val, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons,
         double **Y_train, double **X_val, double **Y_val, double ***W, double **b, double ***a);

void DownloadANN(int epochs, double learning_rate, double initial_range, char *filename, double train_split, int num_train,
                  int num_val, int num_inputs, int num_outputs, int num_hidden_layers, int *num_neurons, 
                  double **X_train, double **Y_train, double **X_val, double **Y_val, double ***W, double **b);

void MakePredictions(double ***W, double **b, int num_inputs, int num_hidden_layers, int *num_neurons, int num_outputs);

#endif