using GLib;

namespace org.esgdev.verceptron {

public class MultilayerPerceptron {
    public double[,] weights { get; private set; }
    public double[] biases { get; private set; }
    // Store LayerDefinition array
    private LayerDefinition[] layer_configs { get; set; }
    // layer_sizes can be derived or stored if frequently used
    public int[] layer_sizes { get; private set; }
    private double learning_rate;

    // Modified Constructor
    public MultilayerPerceptron (LayerDefinition[] layer_configs, double learning_rate = 0.1) {
        this.layer_configs = layer_configs;
        this.learning_rate = learning_rate;

        // Derive layer_sizes from layer_configs
        this.layer_sizes = new int[layer_configs.length];
        for (int i = 0; i < layer_configs.length; i++) {
            this.layer_sizes[i] = layer_configs[i].num_neurons;
        }

        int total_weights = 0;
        int total_biases = 0;
        // Use this.layer_sizes which is now derived
        for (int l = 0; l < this.layer_sizes.length - 1; l++) {
            total_weights += this.layer_sizes[l] * this.layer_sizes[l + 1];
            total_biases += this.layer_sizes[l + 1];
        }

        weights = new double[total_weights, 1];
        biases = new double[total_biases];
        var random = new Rand();
        int weight_index = 0;

        for (int l = 0; l < this.layer_sizes.length - 1; l++) {
            int n_in = this.layer_sizes[l];
            int n_out = this.layer_sizes[l + 1];
            ActivationFunction activation_fn = this.layer_configs[l + 1].activation_function;

            double scale;
            if (activation_fn is ReLUActivation || activation_fn is LeakyReLUActivation) {
                // He Initialization
                scale = Math.sqrt(2.0 / n_in);
            } else if (activation_fn is SigmoidActivation || activation_fn is TanhActivation) {
                // Xavier Initialization
                scale = Math.sqrt(1.0 / n_in);
            } else {
                // Default to small random values
                scale = 0.01;
            }

            for (int i = 0; i < n_in * n_out; i++) {
                weights[weight_index++, 0] = random.next_double() * 2.0 * scale - scale;
            }
        }

        for (int i = 0; i < total_biases; i++) {
            biases[i] = (random.next_double() * 2.0) - 1.0;
        }
    }

    public double[] forward(double[] inputs) {
        double[] activations = inputs;
        int weight_index = 0;
        int bias_index = 0;

        for (int l = 0; l < this.layer_sizes.length - 1; l++) {
            double[] next_activations = new double[this.layer_sizes[l + 1]];
            // Get the activation function for the current layer being computed (layer l+1)
            ActivationFunction current_activation_fn = this.layer_configs[l + 1].activation_function;

            for (int j = 0; j < this.layer_sizes[l + 1]; j++) {
                double weighted_sum = biases[bias_index++];
                for (int i = 0; i < this.layer_sizes[l]; i++) {
                    weighted_sum += activations[i] * weights[weight_index++, 0];
                }
                // Use the layer-specific activation function
                next_activations[j] = current_activation_fn.activate(weighted_sum);
            }
            activations = next_activations;
        }
        return activations;
    }

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
            ActivationFunction current_activation_fn = this.layer_configs[l + 1].activation_function;

            for (int j = 0; j < layer_sizes[l + 1]; j++) {
                double weighted_sum = biases[bias_index++];
                
                for (int i = 0; i < layer_sizes[l]; i++) {
                    weighted_sum += all_activations[current_activation_offset + i] * weights[weight_index++, 0];
                }
                
                // Store the weighted sum (z value)
                z_values[z_offset + j] = weighted_sum;
                
                // Apply activation function and store the result
                all_activations[next_activation_offset + j] = current_activation_fn.activate(weighted_sum);
            }
        }
        
        // Calculate output layer error (dL/dz for output layer)
        int output_layer_idx = layer_sizes.length - 1;
        int output_activation_offset = activation_offsets[output_layer_idx];
        // z_offsets[k] is for layer k+1, so z_offsets[output_layer_idx - 1] is for output_layer_idx
        int output_z_offset = z_offsets[output_layer_idx - 1]; 
        ActivationFunction output_activation_fn = this.layer_configs[output_layer_idx].activation_function;
        
        double[] errors = new double[layer_sizes[output_layer_idx]]; // This will store dL/dz for the output layer
        
        for (int i = 0; i < layer_sizes[output_layer_idx]; i++) {
            double output_activation_val = all_activations[output_activation_offset + i]; // a_i^(L)
            double dL_da_output = output_activation_val - targets[i]; // d(Loss)/da_i^(L)
            double z_val_output = z_values[output_z_offset + i]; // z_i^(L)
            errors[i] = output_activation_fn.backward(z_val_output, dL_da_output); // errors[i] is now dL/dz_i^(L)
        }

        double[] current_deltas = errors; // current_deltas are dL/dz for the current layer being processed in backprop (starts with output layer)
        
        // Backpropagate the error: loop from layer L-1 down to layer 1 (0-indexed: layer_sizes.length-2 down to 1)
        // For l=0 (input layer), we update weights connecting to layer 1, but don't compute deltas for layer 0 itself for further backprop.
        for (int l = layer_sizes.length - 2; l >= 0; l--) { // l is the index of the "from" layer in W_ij (from l to l+1)
            // Reset weight and bias indices for the connection from layer l to layer l+1
            int current_layer_weight_index = 0;
            int current_layer_bias_index = 0;
            for (int prev_layer_idx = 0; prev_layer_idx < l; prev_layer_idx++) {
                current_layer_weight_index += layer_sizes[prev_layer_idx] * layer_sizes[prev_layer_idx + 1];
                current_layer_bias_index += layer_sizes[prev_layer_idx + 1];
            }
            
            double[] dL_da_layer_l = new double[layer_sizes[l]]; // To store dL/da for neurons in layer l
            // Initialize to zero
            for (int i = 0; i < layer_sizes[l]; i++) {
                dL_da_layer_l[i] = 0;
            }
            
            int activation_offset_layer_l = activation_offsets[l]; // Activations a^(l)

            // Update weights and biases connecting layer l to layer l+1
            // And compute dL/da for layer l
            for (int j = 0; j < layer_sizes[l+1]; j++) { // j is neuron index in layer l+1
                // current_deltas[j] is dL/dz_j^(l+1) (delta for neuron j in layer l+1)
                
                // Update bias for neuron j in layer l+1
                biases[current_layer_bias_index + j] -= learning_rate * current_deltas[j];
                
                for (int i = 0; i < layer_sizes[l]; i++) { // i is neuron index in layer l
                    // Weight w_ij connecting neuron i (layer l) to neuron j (layer l+1)
                    int w_idx = current_layer_weight_index + j * layer_sizes[l] + i;
                    
                    double original_weight = weights[w_idx, 0];

                    // Update weight w_ij
                    double weight_gradient = current_deltas[j] * all_activations[activation_offset_layer_l + i]; // dL/dz_j^(l+1) * a_i^(l)
                    weights[w_idx, 0] -= learning_rate * weight_gradient;
                    
                    // Accumulate dL/da_i^(l) = sum_j (dL/dz_j^(l+1) * w_ij^(original))
                    dL_da_layer_l[i] += current_deltas[j] * original_weight;
                }
            }
            
            // If layer l is a hidden layer (l > 0), compute dL/dz for layer l (these become current_deltas for next iteration)
            if (l > 0) {
                double[] dL_dz_layer_l = new double[layer_sizes[l]];
                ActivationFunction activation_fn_l = this.layer_configs[l].activation_function;
                // z_values for layer l are at z_offsets[l-1]
                int z_offset_l = z_offsets[l-1]; 

                for (int i = 0; i < layer_sizes[l]; i++) {
                    double z_value_l = z_values[z_offset_l + i]; // z_i^(l)
                    dL_dz_layer_l[i] = activation_fn_l.backward(z_value_l, dL_da_layer_l[i]); // dL/dz_i^(l)
                }
                current_deltas = dL_dz_layer_l;
            }
            // If l == 0, current_deltas (which are dL/dz for layer 1) are used in this iteration to update weights/biases
            // connecting layer 0 to layer 1. We don't need to compute dL/dz for layer 0 for further backpropagation.
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