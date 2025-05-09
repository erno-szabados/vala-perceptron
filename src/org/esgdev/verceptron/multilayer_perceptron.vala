using GLib;

namespace org.esgdev.verceptron {

public class MultilayerPerceptron {
    public double[,] weights { get; private set; }
    public double[] biases { get; private set; }
    private LayerDefinition[] layer_configs { get; set; }
    public int[] layer_sizes { get; private set; }
    private double learning_rate;
    private ErrorFunction error_function;

    // Adam optimizer parameters
    private double[] m_weights; // First moment vector for weights
    private double[] v_weights; // Second moment vector for weights
    private double[] m_biases;  // First moment vector for biases
    private double[] v_biases;  // Second moment vector for biases
    private double beta1 = 0.9;
    private double beta2 = 0.999;
    private double epsilon = 1e-8;
    private int t = 0; // Time step for bias correction

    public MultilayerPerceptron (LayerDefinition[] layer_configs, double learning_rate = 0.1, ErrorFunction? error_function = null) {
        this.layer_configs = layer_configs;
        this.learning_rate = learning_rate;
        this.error_function = error_function != null ? error_function : new MeanSquaredError();

        this.layer_sizes = new int[layer_configs.length];
        for (int i = 0; i < layer_configs.length; i++) {
            this.layer_sizes[i] = layer_configs[i].num_neurons;
        }

        int total_weights = 0;
        int total_biases = 0;
        for (int l = 0; l < this.layer_sizes.length - 1; l++) {
            total_weights += this.layer_sizes[l] * this.layer_sizes[l + 1];
            total_biases += this.layer_sizes[l + 1];
        }

        weights = new double[total_weights, 1]; // Storing weights in a way that can be accessed 1D
        biases = new double[total_biases];
        
        // Initialize Adam moment vectors
        this.m_weights = new double[total_weights];
        this.v_weights = new double[total_weights];
        this.m_biases = new double[total_biases];
        this.v_biases = new double[total_biases];
        // t is already initialized to 0 by default for int class member

        var random = new Rand();
        int weight_idx_init = 0; // Renamed to avoid conflict

        for (int l = 0; l < this.layer_sizes.length - 1; l++) {
            int n_in = this.layer_sizes[l];
            int n_out = this.layer_sizes[l + 1];
            ActivationFunction activation_fn = this.layer_configs[l + 1].activation_function;

            double scale;
            if (activation_fn is ReLUActivation || activation_fn is LeakyReLUActivation) {
                scale = Math.sqrt(2.0 / n_in);
            } else if (activation_fn is SigmoidActivation || activation_fn is TanhActivation) {
                scale = Math.sqrt(1.0 / n_in);
            } else {
                scale = 0.01;
            }

            for (int i = 0; i < n_in * n_out; i++) {
                weights[weight_idx_init++, 0] = random.next_double() * 2.0 * scale - scale;
            }
        }

        for (int i = 0; i < total_biases; i++) {
            // Consider initializing biases to zero or small constants for Adam
            biases[i] = 0.0; // Or (random.next_double() * 0.02) - 0.01; 
        }
    }

    public double[] forward(double[] inputs) {
        double[] activations = inputs;

        for (int l = 0; l < this.layer_sizes.length - 1; l++) {
            double[] next_activations = new double[this.layer_sizes[l + 1]];
            ActivationFunction current_activation_fn = this.layer_configs[l + 1].activation_function;
            int current_layer_weight_start_index = 0;
            for(int k=0; k<l; k++){
                current_layer_weight_start_index += layer_sizes[k] * layer_sizes[k+1];
            }
            int current_layer_bias_start_index = 0;
            for(int k=0; k<l; k++){
                current_layer_bias_start_index += layer_sizes[k+1];
            }

            for (int j = 0; j < this.layer_sizes[l + 1]; j++) { // Neuron in next layer
                double weighted_sum = biases[current_layer_bias_start_index + j];
                for (int i = 0; i < this.layer_sizes[l]; i++) { // Neuron in current layer / input to this connection
                    // Correct weight indexing for forward pass
                    int w_idx = current_layer_weight_start_index + j * this.layer_sizes[l] + i;
                    weighted_sum += activations[i] * weights[w_idx, 0];
                }
                next_activations[j] = current_activation_fn.activate(weighted_sum);
            }
            activations = next_activations;
        }
        return activations;
    }

    public void train(double[] inputs, double[] targets) {
        this.t++; // Increment Adam time step

        // Create an array to store all activations for all layers
        int total_activations = 0;
        for (int i = 0; i < layer_sizes.length; i++) {
            total_activations += layer_sizes[i];
        }
        double[] all_activations = new double[total_activations];
        int[] activation_offsets = new int[layer_sizes.length];
        
        int offset = 0;
        for (int i = 0; i < layer_sizes.length; i++) {
            activation_offsets[i] = offset;
            offset += layer_sizes[i];
        }
        
        for (int i = 0; i < inputs.length; i++) {
            all_activations[i] = inputs[i];
        }
        
        int total_z_values = total_activations - layer_sizes[0];
        double[] z_values = new double[total_z_values];
        int[] z_offsets = new int[layer_sizes.length - 1];
        
        offset = 0;
        for (int i = 0; i < layer_sizes.length - 1; i++) {
            z_offsets[i] = offset;
            offset += layer_sizes[i + 1];
        }
        
        // Recalculate global weight and bias indices for forward pass in train
        int global_weight_idx = 0;
        int global_bias_idx = 0;
        
        for (int l = 0; l < layer_sizes.length - 1; l++) {
            int current_activation_offset = activation_offsets[l];
            int next_activation_offset = activation_offsets[l + 1];
            int current_z_offset = z_offsets[l]; // z_values for layer l+1
            ActivationFunction current_activation_fn = this.layer_configs[l + 1].activation_function;

            for (int j = 0; j < layer_sizes[l + 1]; j++) { // Neuron in layer l+1
                double weighted_sum = biases[global_bias_idx + j]; 
                
                for (int i = 0; i < layer_sizes[l]; i++) { // Neuron in layer l
                    int w_flat_idx = global_weight_idx + j * layer_sizes[l] + i;
                    weighted_sum += all_activations[current_activation_offset + i] * weights[w_flat_idx, 0];
                }
                z_values[current_z_offset + j] = weighted_sum;
                all_activations[next_activation_offset + j] = current_activation_fn.activate(weighted_sum);
            }
            global_weight_idx += layer_sizes[l] * layer_sizes[l+1];
            global_bias_idx += layer_sizes[l+1];
        }

        int output_layer_idx = layer_sizes.length - 1;
        int output_activation_offset = activation_offsets[output_layer_idx];
        int output_z_offset = z_offsets[output_layer_idx - 1]; 
        ActivationFunction output_activation_fn = this.layer_configs[output_layer_idx].activation_function;
        
        double[] errors = new double[layer_sizes[output_layer_idx]];
        
        for (int i = 0; i < layer_sizes[output_layer_idx]; i++) {
            double output_activation_val = all_activations[output_activation_offset + i];
            double dL_da_output = this.error_function.backwards(targets[i], output_activation_val);
            double z_val_output = z_values[output_z_offset + i];
            errors[i] = output_activation_fn.backward(z_val_output, dL_da_output);
        }

        double[] current_deltas = errors; 
        
        for (int l = layer_sizes.length - 2; l >= 0; l--) {
            int current_layer_num_neurons = layer_sizes[l];
            int next_layer_num_neurons = layer_sizes[l+1];
            
            int activation_offset_layer_l = activation_offsets[l];
            
            int weight_start_index_for_layer = 0;
            int bias_start_index_for_layer = 0;
            for (int prev_l = 0; prev_l < l; prev_l++) {
                weight_start_index_for_layer += layer_sizes[prev_l] * layer_sizes[prev_l + 1];
                bias_start_index_for_layer += layer_sizes[prev_l + 1];
            }

            double[] dL_da_layer_l = new double[current_layer_num_neurons];

            for (int j = 0; j < next_layer_num_neurons; j++) { 
                double delta_j_next_layer = current_deltas[j]; 

                double bias_gradient = delta_j_next_layer;
                int bias_idx = bias_start_index_for_layer + j;

                m_biases[bias_idx] = beta1 * m_biases[bias_idx] + (1 - beta1) * bias_gradient;
                v_biases[bias_idx] = beta2 * v_biases[bias_idx] + (1 - beta2) * (bias_gradient * bias_gradient);
                double m_hat_bias = m_biases[bias_idx] / (1 - Math.pow(beta1, this.t));
                double v_hat_bias = v_biases[bias_idx] / (1 - Math.pow(beta2, this.t));
                biases[bias_idx] -= learning_rate * m_hat_bias / (Math.sqrt(v_hat_bias) + epsilon);

                for (int i = 0; i < current_layer_num_neurons; i++) { 
                    int w_idx = weight_start_index_for_layer + j * current_layer_num_neurons + i;
                    double original_weight = weights[w_idx, 0]; 

                    double weight_gradient = delta_j_next_layer * all_activations[activation_offset_layer_l + i];
                    
                    m_weights[w_idx] = beta1 * m_weights[w_idx] + (1 - beta1) * weight_gradient;
                    v_weights[w_idx] = beta2 * v_weights[w_idx] + (1 - beta2) * (weight_gradient * weight_gradient);
                    double m_hat_weight = m_weights[w_idx] / (1 - Math.pow(beta1, this.t));
                    double v_hat_weight = v_weights[w_idx] / (1 - Math.pow(beta2, this.t));
                    weights[w_idx, 0] -= learning_rate * m_hat_weight / (Math.sqrt(v_hat_weight) + epsilon);
                    
                    dL_da_layer_l[i] += delta_j_next_layer * original_weight;
                }
            }
            
            if (l > 0) { 
                double[] dL_dz_layer_l = new double[current_layer_num_neurons];
                ActivationFunction activation_fn_l = this.layer_configs[l].activation_function;
                int z_offset_l = z_offsets[l-1]; 

                for (int i = 0; i < current_layer_num_neurons; i++) {
                    double z_value_l = z_values[z_offset_l + i];
                    dL_dz_layer_l[i] = activation_fn_l.backward(z_value_l, dL_da_layer_l[i]);
                }
                current_deltas = dL_dz_layer_l;
            }
        }
    }

    public void fit(double[,] x_train, double[] y_train, int epochs, bool verbose = false) {
        int samples = x_train.length[0];
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double sum_error = 0;
            
            for (int sample_idx = 0; sample_idx < samples; sample_idx++) {
                double[] input = new double[layer_sizes[0]];
                for (int j = 0; j < input.length; j++) {
                    input[j] = x_train[sample_idx, j];
                }
                
                double[] target = new double[layer_sizes[layer_sizes.length-1]];
                target[0] = y_train[sample_idx]; 
                
                this.train(input, target); 
                
                if (verbose) {
                    double[] output = this.forward(input); 
                    for (int k = 0; k < output.length; k++) {
                        sum_error += this.error_function.compute(target[k], output[k]);
                    }
                }
            }
            
            if (verbose && epoch % 100 == 0) {
                double avg_error = sum_error / samples;
                stdout.printf("Epoch %d: Error = %.6f\n", epoch, avg_error);
            }
        }
    }
    
    public double[] predict_batch(double[,] x) {
        int samples = x.length[0];
        int output_size = layer_sizes[layer_sizes.length-1]; // This will now be used
        
        double[] predictions = new double[samples * output_size]; 
        
        for (int i = 0; i < samples; i++) {
            double[] input = new double[layer_sizes[0]];
            for (int j = 0; j < input.length; j++) {
                input[j] = x[i, j];
            }
            double[] output_for_sample = this.forward(input); // output_for_sample will have 'output_size' elements
            
            // Copy all output values for the current sample into the predictions array
            for (int k = 0; k < output_size; k++) {
                predictions[i * output_size + k] = output_for_sample[k];
            }
        }
        return predictions;
    }
}

} // end namespace