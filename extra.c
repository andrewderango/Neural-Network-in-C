       // if (ep%100==0) {
        //     double ***a_eval_train = malloc((num_hidden_layers + 1) * sizeof(double **));
        //     for (int layer = 0; layer <= num_hidden_layers; layer++) {
        //         int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        //         a_eval_train[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        //         for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
        //             a_eval_train[layer][neuron] = malloc(num_train * sizeof(double));
        //         }
        //     }
        //     ForwardPass(num_train, num_inputs, num_outputs, num_hidden_layers, num_neurons,
        //                 X_train, Y_train,
        //                 W, b, a_eval_train);

        //     int correct_predictions = 0;
        //     for (int i = 0; i < num_train; i++) {
        //         int all_correct = 1; // Assume all outputs are correct for this example
        //         for (int j = 0; j < num_outputs; j++) {
        //             if ((a_eval_train[j][0][i] >= 0.5 && Y_train[i][j] == 0) || (a_eval_train[j][0][i] < 0.5 && Y_train[i][j] == 1)) {
        //                 all_correct = 0; // Found an incorrect prediction for this example
        //                 break;
        //             }
        //         }
        //         correct_predictions += all_correct;
        //     }

        //     double accuracy_train = (double)correct_predictions / num_train * 100.0;

        //     double sum_squared_diff = 0.0;
        //     for (int i = 0; i < num_train; i++)
        //     {
        //         for (int j = 0; j < num_outputs; j++)
        //         {
        //             double diff = Y_train[i][j] - a_eval_train[j][0][i]; // Transpose a_eval_train by switching indices
        //             sum_squared_diff += diff * diff;
        //         }
        //     }

        //     // Calculate the cost and divide by num_train
        //     double cost_train = sum_squared_diff / num_train;

        //     double ***a_val = malloc((num_hidden_layers + 1) * sizeof(double **));
        //     for (int layer = 0; layer <= num_hidden_layers; layer++) {
        //         int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        //         a_val[layer] = malloc(num_neurons_current_layer * sizeof(double *));
        //         for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
        //             a_val[layer][neuron] = malloc(num_val * sizeof(double));
        //         }
        //     }

        //     ForwardPass(num_val, num_inputs, num_outputs, num_hidden_layers, num_neurons,
        //                 X_val, Y_val,
        //                 W, b, a_val);

        //     correct_predictions = 0;
        //     for (int i = 0; i < num_val; i++)
        //     {
        //         int all_correct = 1; // Assume all outputs are correct for this example
        //         for (int j = 0; j < num_outputs; j++)
        //         {
        //             if ((a_val[j][0][i] >= 0.5 && Y_val[i][j] == 0) || (a_val[j][0][i] < 0.5 && Y_val[i][j] == 1))
        //             {
        //                 all_correct = 0; // Found an incorrect prediction for this example
        //                 break;
        //             }
        //         }
        //         correct_predictions += all_correct;
        //     }

        //     double accuracy_val = (double)correct_predictions / num_val * 100.0;

        //     sum_squared_diff = 0.0;
        //     for (int i = 0; i < num_val; i++)
        //     {
        //         for (int j = 0; j < num_outputs; j++)
        //         {
        //             double diff = Y_val[i][j] - a_val[j][0][i]; // Transpose a_eval_train by switching indices
        //             sum_squared_diff += diff * diff;
        //         }
        //     }

        //     // Calculate the cost and divide by num_train
        //     double cost_val = sum_squared_diff / num_val;

        //     printf("Epoch %d:\n", ep);
        //     printf("Train Cost      %lf,    Accuracy: %.2f%%\n",cost_train, accuracy_train);
        //     printf("Validation Cost %lf,    Accuracy: %.2f%%\n\n",cost_val, accuracy_val);


        //     for (int layer = 0; layer <= num_hidden_layers; layer++) {
        //         int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        //         for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
        //             free(a_val[layer][neuron]);
        //         }
        //         free(a_val[layer]);
        //     }
        //     free(a_val);
        //     for (int layer = 0; layer <= num_hidden_layers; layer++) {
        //         int num_neurons_current_layer = (layer == num_hidden_layers) ? num_outputs : num_neurons[layer];
        //         for (int neuron = 0; neuron < num_neurons_current_layer; neuron++) {
        //             free(a_eval_train[layer][neuron]);
        //         }
        //         free(a_eval_train[layer]);
        //     }
        //     free(a_eval_train);
        // }