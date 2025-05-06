using GLib;
using org.esgdev.verceptron;

void test_constructor_initialization () {
    int[] layer_sizes = { 3, 2, 1 };
    double learning_rate = 0.1;
    var mlp = new MultilayerPerceptron (layer_sizes, learning_rate);

    // Check layer sizes
    assert (mlp.layer_sizes.length == 3);

    // Check weights and biases are initialized
    int total_weights = 3 * 2 + 2 * 1; // 6 + 2 = 8
    int total_biases = 2 + 1;          // 3
    assert (mlp.weights.length[0] == total_weights);
    assert (mlp.biases.length == total_biases);
}

void test_forward_propagation () {
    int[] layer_sizes = { 2, 2, 1 };
    var mlp = new MultilayerPerceptron (layer_sizes);

    // Test with simple inputs
    double[] inputs = { 1.0, 0.5 };
    double[] outputs = mlp.forward (inputs);

    // Check output size matches the last layer
    assert (outputs.length == 1);
    
    // Optional: Print output for verification
    // stdout.printf("Output: %f\n", outputs[0]);
}

void test_backpropagation_xor() {
    // Create a network with 2 inputs, 3 hidden neurons, and 1 output
    int[] layer_sizes = { 2, 3, 1 };
    var mlp = new MultilayerPerceptron(layer_sizes, 0.1);
    
    // XOR training data
    double[,] x_train = {
        { 0.0, 0.0 },
        { 0.0, 1.0 },
        { 1.0, 0.0 },
        { 1.0, 1.0 }
    };
    
    double[] y_train = { 0.0, 1.0, 1.0, 0.0 };
    
    // Train the network
    mlp.fit(x_train, y_train, 2000, false);
    
    // Test predictions
    double[] predictions = mlp.predict_batch(x_train);
    
    // Verify that predictions are close to expected values
    // We use a tolerance because neural network training is approximate
    double tolerance = 0.2;
    
    for (int i = 0; i < 4; i++) {
        double prediction = predictions[i];
        double expected = y_train[i];
        
        // Print the values for debugging
        stdout.printf("Input: [%.1f, %.1f], Expected: %.1f, Predicted: %.6f\n",
                     x_train[i, 0], x_train[i, 1], expected, prediction);
        
        // Assert that prediction is close to expected
        assert(Math.fabs(prediction - expected) < tolerance);
    }
}

void test_backpropagation_binary_classification() {
    // Create a simple network for binary classification
    int[] layer_sizes = { 2, 4, 1 };
    var mlp = new MultilayerPerceptron(layer_sizes, 0.05);
    
    // Create a simple linearly separable dataset
    // Points above the line y = x will be class 1, below will be class 0
    double[,] x_train = {
        { 0.0, 0.1 }, // Class 1 (0.1 > 0.0)
        { 0.1, 0.3 }, // Class 1 (0.3 > 0.1)
        { 0.5, 0.6 }, // Class 1 (0.6 > 0.5)
        { 0.7, 0.9 }, // Class 1 (0.9 > 0.7)
        { 0.1, 0.0 }, // Class 0 (0.0 < 0.1)
        { 0.3, 0.2 }, // Class 0 (0.2 < 0.3)
        { 0.6, 0.4 }, // Class 0 (0.4 < 0.6)
        { 0.9, 0.7 }  // Class 0 (0.7 < 0.9)
    };
    
    double[] y_train = { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
    
    // Train the network with verbose output
    mlp.fit(x_train, y_train, 1000, true);
    
    // Test predictions
    double[] predictions = mlp.predict_batch(x_train);
    
    // Define a tolerance for binary classification
    double tolerance = 0.2;
    bool all_correct = true;
    
    stdout.printf("\nBinary Classification Results:\n");
    for (int i = 0; i < 8; i++) {
        double prediction = predictions[i];
        double expected = y_train[i];
        
        // Print the values
        stdout.printf("Input: [%.1f, %.1f], Expected: %.1f, Predicted: %.6f\n",
                     x_train[i, 0], x_train[i, 1], expected, prediction);
        
        // Record if any prediction is too far from expected
        if (Math.fabs(prediction - expected) >= tolerance) {
            all_correct = false;
        }
    }
    
    // Test additional points not in the training set
    double[,] x_test = {
        { 0.2, 0.3 }, // Class 1 (0.3 > 0.2)
        { 0.4, 0.2 }, // Class 0 (0.2 < 0.4)
    };
    
    double[] expected_test = { 1.0, 0.0 };
    double[] test_predictions = mlp.predict_batch(x_test);
    
    stdout.printf("\nTest Points:\n");
    for (int i = 0; i < 2; i++) {
        double prediction = test_predictions[i];
        double expected = expected_test[i];
        
        stdout.printf("Test Input: [%.1f, %.1f], Expected: %.1f, Predicted: %.6f\n",
                     x_test[i, 0], x_test[i, 1], expected, prediction);
        
        // Check if prediction is correct within tolerance
        if (Math.fabs(prediction - expected) >= tolerance) {
            all_correct = false;
        }
    }
    
    // Assert that all predictions are correct
    assert(all_correct);
}

int main (string[] args) {
    Test.init (ref args);

    // Add test cases
    Test.add_func ("/multilayer_perceptron/constructor_initialization", test_constructor_initialization);
    Test.add_func ("/multilayer_perceptron/forward_propagation", test_forward_propagation);
    Test.add_func ("/multilayer_perceptron/backpropagation_xor", test_backpropagation_xor);
    Test.add_func ("/multilayer_perceptron/backpropagation_binary_classification", test_backpropagation_binary_classification);

    return Test.run ();
}