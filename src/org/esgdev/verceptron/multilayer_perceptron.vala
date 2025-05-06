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
        
        // 2. Backward pass to compute gradients
        //double[,] weight_gradients = new double[weights.length[0], 1];
        //double[] bias_gradients = new double[biases.length];
        
        // Calculate output layer error
        int output_layer = layer_sizes.length - 1;
        int output_offset = activation_offsets[output_layer];
        int last_z_offset = z_offsets[output_layer - 1];
        
        double[] errors = new double[layer_sizes[output_layer]];
        
        for (int i = 0; i < layer_sizes[output_layer]; i++) {
            double output = all_activations[output_offset + i];
            errors[i] = (output - targets[i]) * relu_derivative(z_values[last_z_offset + i]);
        }
        
        // TODO: Propagate errors backwards and update weights
        // Continue with backpropagation algorithm...
    }
}

} // end namespace