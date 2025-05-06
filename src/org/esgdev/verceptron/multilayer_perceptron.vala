using GLib;

namespace org.esgdev.verceptron {

public class MultilayerPerceptron {
    // Make properties public for testing
    public double[,] weights { get; private set; } // 2D array for weights
    public double[] biases { get; private set; }   // 1D array for biases
    public int[] layer_sizes { get; private set; } // Number of neurons in each layer
    private double learning_rate;

    // Constructor
    public MultilayerPerceptron (int[] layer_sizes, double learning_rate = 0.1) {
        this.layer_sizes = layer_sizes;
        this.learning_rate = learning_rate;

        // Calculate total number of weights and biases
        int total_weights = 0;
        int total_biases = 0;
        for (int l = 0; l < layer_sizes.length - 1; l++) {
            total_weights += layer_sizes[l] * layer_sizes[l + 1];
            total_biases += layer_sizes[l + 1];
        }

        // Initialize weights and biases
        weights = new double[total_weights, 1]; // Flattened 2D array
        biases = new double[total_biases];
        var random = new Rand();

        // Fill weights and biases with random values
        for (int i = 0; i < total_weights; i++) {
            weights[i, 0] = (random.next_double() * 2.0) - 1.0; // Random [-1, 1)
        }
        for (int i = 0; i < total_biases; i++) {
            biases[i] = (random.next_double() * 2.0) - 1.0; // Random [-1, 1)
        }
    }

    // Forward propagation
    public double[] forward(double[] inputs) {
        double[] activations = inputs;
        int weight_index = 0;
        int bias_index = 0;

        for (int l = 0; l < layer_sizes.length - 1; l++) {
            double[] next_activations = new double[layer_sizes[l + 1]];

            for (int j = 0; j < layer_sizes[l + 1]; j++) {
                double weighted_sum = biases[bias_index++];
                for (int i = 0; i < layer_sizes[l]; i++) {
                    weighted_sum += activations[i] * weights[weight_index++, 0];
                }
                next_activations[j] = relu(weighted_sum); // Using ReLU as default activation
            }

            activations = next_activations;
        }

        return activations;
    }

    // Activation function (ReLU)
    private double relu(double x) {
        return (x > 0) ? x : 0;
    }
    
    // Derivative of ReLU
    private double relu_derivative(double x) {
        return (x > 0) ? 1.0 : 0.0;
    }

    // Backpropagation and training
    public void train(double[] inputs, double[] targets) {
        // 1. Use arrays and offsets instead of jagged arrays 
        // Create an array to store all activations for all layers
        int total_activations = 0;
        for (int i = 0; i < layer_sizes.length; i++) {
            total_activations += layer_sizes[i];
        }
        double[] all_activations = new double[total_activations];
        int[] activation_offsets = new int[layer_sizes.length];
        
        // Calculate offsets for each layer's activations in the array
        int offset = 0;
        for (int i = 0; i < layer_sizes.length; i++) {
            activation_offsets[i] = offset;
            offset += layer_sizes[i];
        }
        
        // Copy input activations to the first layer
        for (int i = 0; i < inputs.length; i++) {
            all_activations[i] = inputs[i];
        }
        
        // Store weighted sums for derivative calculation
        int total_z_values = total_activations - layer_sizes[0]; // No z values for input layer
        double[] z_values = new double[total_z_values];
        int[] z_offsets = new int[layer_sizes.length - 1];
        
        // Calculate offsets for z values
        offset = 0;
        for (int i = 0; i < layer_sizes.length - 1; i++) {
            z_offsets[i] = offset;
            offset += layer_sizes[i + 1];
        }
        
        // Forward pass with storage of intermediate values
        int weight_index = 0;
        int bias_index = 0;
        
        for (int l = 0; l < layer_sizes.length - 1; l++) {
            int current_activation_offset = activation_offsets[l];
            int next_activation_offset = activation_offsets[l + 1];
            int z_offset = z_offsets[l];
            
            for (int j = 0; j < layer_sizes[l + 1]; j++) {
                double weighted_sum = biases[bias_index++];
                
                for (int i = 0; i < layer_sizes[l]; i++) {
                    weighted_sum += all_activations[current_activation_offset + i] * weights[weight_index++, 0];
                }
                
                // Store the weighted sum (z value)
                z_values[z_offset + j] = weighted_sum;
                
                // Apply activation function and store the result
                all_activations[next_activation_offset + j] = relu(weighted_sum);
            }
        }
        
        // Calculate output layer error
        int output_layer = layer_sizes.length - 1;
        int output_offset = activation_offsets[output_layer];
        int last_z_offset = z_offsets[output_layer - 1];
        
        double[] errors = new double[layer_sizes[output_layer]];
        
        for (int i = 0; i < layer_sizes[output_layer]; i++) {
            double output = all_activations[output_offset + i];
            errors[i] = (output - targets[i]) * relu_derivative(z_values[last_z_offset + i]);
        }

        // Arrays to store deltas for each layer
        double[] current_deltas = errors;
        
        // Backpropagate the error and update weights
        for (int l = layer_sizes.length - 2; l >= 0; l--) {
            // Reset weight and bias indices for this layer
            weight_index = 0;
            bias_index = 0;
            
            // Skip to the weights/biases for the current layer
            for (int prev_layer = 0; prev_layer < l; prev_layer++) {
                weight_index += layer_sizes[prev_layer] * layer_sizes[prev_layer + 1];
                bias_index += layer_sizes[prev_layer + 1];
            }
            
            // Prepare deltas for the previous layer (if not input layer)
            double[] next_deltas = null;
            if (l > 0) {
                next_deltas = new double[layer_sizes[l]];
                // Initialize to zero
                for (int i = 0; i < layer_sizes[l]; i++) {
                    next_deltas[i] = 0;
                }
            }
            
            // Current layer activation offset
            int current_activation_offset = activation_offsets[l];
            int current_z_offset = (l > 0) ? z_offsets[l-1] : 0;
            
            // Update weights and biases for current layer
            for (int j = 0; j < layer_sizes[l+1]; j++) {
                // Update bias with the delta
                biases[bias_index + j] -= learning_rate * current_deltas[j];
                
                // Update weights for this neuron
                for (int i = 0; i < layer_sizes[l]; i++) {
                    // Calculate weight index - CORRECTED
                    int w_idx = weight_index + j * layer_sizes[l] + i;
                    
                    // Store original weight for delta propagation
                    double original_weight_for_delta_prop = weights[w_idx, 0];

                    // Update weight
                    double weight_update = learning_rate * current_deltas[j] * 
                                         all_activations[current_activation_offset + i];
                    weights[w_idx, 0] -= weight_update;
                    
                    // Propagate error to previous layer (if not input layer)
                    if (l > 0) {
                        // Use original weight for propagating error - CORRECTED
                        next_deltas[i] += current_deltas[j] * original_weight_for_delta_prop;
                    }
                }
            }
            
            // Apply derivative for next layer's deltas
            if (l > 0) {
                for (int i = 0; i < layer_sizes[l]; i++) {
                    next_deltas[i] *= relu_derivative(z_values[current_z_offset + i]);
                }
                // Set the calculated deltas as current for the next iteration
                current_deltas = next_deltas;
            }
        }
    }
    
    // Train the network for multiple epochs
    public void fit(double[,] x_train, double[] y_train, int epochs, bool verbose = false) {
        int samples = x_train.length[0];
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double sum_error = 0;
            
            // Train on all samples
            for (int i = 0; i < samples; i++) {
                // Extract input features
                double[] input = new double[layer_sizes[0]];
                for (int j = 0; j < input.length; j++) {
                    input[j] = x_train[i, j];
                }
                
                // Get target
                double[] target = new double[layer_sizes[layer_sizes.length-1]];
                // Assuming single output for now, can be extended for multi-output
                target[0] = y_train[i];
                
                // Train on this sample
                train(input, target);
                
                // Calculate error for progress reporting
                if (verbose) {
                    double[] output = forward(input);
                    sum_error += Math.pow(output[0] - target[0], 2);
                }
            }
            
            // Report progress if verbose
            if (verbose && epoch % 100 == 0) {
                double mse = sum_error / samples;
                stdout.printf("Epoch %d: MSE = %.6f\n", epoch, mse);
            }
        }
    }
    
    // Predict output for a batch of inputs
    public double[] predict_batch(double[,] x) {
        int samples = x.length[0];
        int output_size = layer_sizes[layer_sizes.length-1];
        double[] predictions = new double[samples * output_size];
        
        for (int i = 0; i < samples; i++) {
            // Extract features
            double[] input = new double[layer_sizes[0]];
            for (int j = 0; j < input.length; j++) {
                input[j] = x[i, j];
            }
            
            // Get prediction
            double[] output = forward(input);
            
            // Store prediction
            for (int j = 0; j < output_size; j++) {
                predictions[i * output_size + j] = output[j];
            }
        }
        
        return predictions;
    }
}

} // end namespace