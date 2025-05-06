using GLib;

public class MultilayerPerceptron {
    private double[,] weights; // 2D array for weights
    private double[] biases;   // 1D array for biases
    private int[] layer_sizes; // Number of neurons in each layer
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
        var random = new Rand ();

        // Fill weights and biases with random values
        for (int i = 0; i < total_weights; i++) {
            weights[i, 0] = (random.next_double () * 2.0) - 1.0; // Random [-1, 1)
        }
        for (int i = 0; i < total_biases; i++) {
            biases[i] = (random.next_double () * 2.0) - 1.0; // Random [-1, 1)
        }
    }

    // Forward propagation
    public double[] forward (double[] inputs) {
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
                next_activations[j] = relu (weighted_sum); // Using ReLU as default activation
            }

            activations = next_activations;
        }

        return activations;
    }

    // Activation function (ReLU)
    private double relu (double x) {
        return (x > 0) ? x : 0;
    }

    // Backpropagation (to be implemented)
    public void train (double[] inputs, double[] targets) {
        // Placeholder for backpropagation logic
    }
}