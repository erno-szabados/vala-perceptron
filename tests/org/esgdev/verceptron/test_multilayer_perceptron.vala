using GLib;
using org.esgdev.verceptron;

void test_constructor_initialization () {
    var layer_configs = new LayerDefinition[] {
        new LayerDefinition(3, new ReLUActivation()),
        new LayerDefinition(2, new SigmoidActivation()),
        new LayerDefinition(1, new SigmoidActivation())
    };
    double learning_rate = 0.1;
    var mlp = new MultilayerPerceptron(layer_configs, learning_rate, new MeanSquaredError(), 42);

    // Check layer sizes
    assert(mlp.layer_sizes.length == 3);

    // Check weights and biases are initialized
    int total_weights = 3 * 2 + 2 * 1; // 6 + 2 = 8
    int total_biases = 2 + 1;          // 3
    assert(mlp.weights.length[0] == total_weights);
    assert(mlp.biases.length == total_biases);
}

void test_forward_propagation () {
    var layer_configs = new LayerDefinition[] {
        new LayerDefinition(2, new ReLUActivation()),
        new LayerDefinition(2, new ReLUActivation()),
        new LayerDefinition(1, new SigmoidActivation())
    };
    var mlp = new MultilayerPerceptron(layer_configs, 0.1, new MeanSquaredError());

    // Test with simple inputs
    double[] inputs = { 1.0, 0.5 };
    double[] outputs = mlp.forward(inputs);

    // Check output size matches the last layer
    assert(outputs.length == 1);
    
    // Optional: Print output for verification
    // stdout.printf("Output: %f\n", outputs[0]);
}

// This test is expected to fail on about 1% of the runs on random seeds
// due to the nature of the XOR problem and the random initialization.
// The fixed seed is used to ensure reproducibility.
void test_backpropagation_xor() {
    var layer_configs = new LayerDefinition[] {
        new LayerDefinition(2, new LeakyReLUActivation()),
        new LayerDefinition(5, new LeakyReLUActivation()),
        new LayerDefinition(1, new SigmoidActivation())
    };
    var mlp = new MultilayerPerceptron(layer_configs, 0.02, new BinaryCrossEntropy(), 42);
    
    // XOR training data
    double[,] x_train = {
        { 0.0, 0.0 },
        { 0.0, 1.0 },
        { 1.0, 0.0 },
        { 1.0, 1.0 }
    };
    
    double[] y_train = { 0.0, 1.0, 1.0, 0.0 };
    
    // Train the network
    mlp.fit(x_train, y_train, 3000, true);
    
    // Test predictions
    double[] predictions = mlp.predict_batch(x_train);
    
    // Verify that predictions are close to expected values
    double tolerance = 0.2;
    
    for (int i = 0; i < 4; i++) {
        double prediction = predictions[i];
        double expected = y_train[i];
        
        stdout.printf("Input: [%.1f, %.1f], Expected: %.1f, Predicted: %.6f\n",
                     x_train[i, 0], x_train[i, 1], expected, prediction);
        
        assert(Math.fabs(prediction - expected) < tolerance);
    }
}

void test_backpropagation_binary_classification() {
    var layer_configs = new LayerDefinition[] {
        new LayerDefinition(2, new LeakyReLUActivation()),
        new LayerDefinition(4, new LeakyReLUActivation()),
        new LayerDefinition(1, new SigmoidActivation())
    };
    var mlp = new MultilayerPerceptron(layer_configs, 0.02, new BinaryCrossEntropy(), 42);
    
    // Create a simple linearly separable dataset
    double[,] x_train = {
        { 0.0, 0.1 },
        { 0.1, 0.3 },
        { 0.5, 0.6 },
        { 0.7, 0.9 },
        { 0.1, 0.0 },
        { 0.3, 0.2 },
        { 0.6, 0.4 },
        { 0.9, 0.7 }
    };
    
    double[] y_train = { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
    
    // Train the network with verbose output
    mlp.fit(x_train, y_train, 2000, true);
    
    // Test predictions
    double[] predictions = mlp.predict_batch(x_train);
    
    double tolerance = 0.2;
    bool all_correct = true;
    
    stdout.printf("\nBinary Classification Results:\n");
    for (int i = 0; i < 8; i++) {
        double prediction = predictions[i];
        double expected = y_train[i];
        
        stdout.printf("Input: [%.1f, %.1f], Expected: %.1f, Predicted: %.6f\n",
                     x_train[i, 0], x_train[i, 1], expected, prediction);
        
        if (Math.fabs(prediction - expected) >= tolerance) {
            all_correct = false;
        }
    }
    
    double[,] x_test = {
        { 0.2, 0.3 },
        { 0.4, 0.2 },
    };
    
    double[] expected_test = { 1.0, 0.0 };
    double[] test_predictions = mlp.predict_batch(x_test);
    
    stdout.printf("\nTest Points:\n");
    for (int i = 0; i < 2; i++) {
        double prediction = test_predictions[i];
        double expected = expected_test[i];
        
        stdout.printf("Test Input: [%.1f, %.1f], Expected: %.1f, Predicted: %.6f\n",
                     x_test[i, 0], x_test[i, 1], expected, prediction);
        
        if (Math.fabs(prediction - expected) >= tolerance) {
            all_correct = false;
        }
    }
    
    assert(all_correct);
}

// Test linear regression: y = 2x + 1
void test_linear_regression() {
    var layer_configs = new LayerDefinition[] {
        new LayerDefinition(1, new IdentityActivation()), // Input layer
        new LayerDefinition(1, new IdentityActivation())  // Output layer
    };
    var mlp = new MultilayerPerceptron(layer_configs, 0.05, new MeanSquaredError(), 42);

    // Training data: y = 2x + 1
    double[,] x_train = {
        { 0.0 },
        { 1.0 },
        { 2.0 },
        { 3.0 },
        { 4.0 }
    };
    double[] y_train = { 1.0, 3.0, 5.0, 7.0, 9.0 };

    mlp.fit(x_train, y_train, 200, true);

    // Test predictions
    double[] predictions = mlp.predict_batch(x_train);
    double tolerance = 0.2;
    for (int i = 0; i < y_train.length; i++) {
        double prediction = predictions[i];
        double expected = y_train[i];
        stdout.printf("Input: %.1f, Expected: %.1f, Predicted: %.3f\n", x_train[i,0], expected, prediction);
        assert(Math.fabs(prediction - expected) < tolerance);
    }
}

// Test function approximation: y = sin(x)
void test_function_approximation_sin() {
    // Network: 1 input, 1 hidden layer (Tanh), 1 output (Identity)
    var layer_configs = new LayerDefinition[] {
        new LayerDefinition(1, new IdentityActivation()),
        new LayerDefinition(8, new TanhActivation()),
        new LayerDefinition(1, new IdentityActivation())
    };
    var mlp = new MultilayerPerceptron(layer_configs, 0.001, new MeanSquaredError(), 42);

    // Training data: x in [0, 2pi], y = sin(x)
    int n_samples = 50;
    double[,] x_train = new double[n_samples, 1];
    double[] y_train = new double[n_samples];
    for (int i = 0; i < n_samples; i++) {
        double x = (2 * Math.PI) * i / (n_samples - 1);
        x_train[i, 0] = x;
        y_train[i] = Math.sin(x);
    }

    mlp.fit(x_train, y_train, 2000, true);

    // Test predictions on training data
    double[] predictions = mlp.predict_batch(x_train);
    double tolerance = 0.15;
    // Compute regression metrics
    double mse = RegressionMetrics.mse(y_train, predictions);
    double mae = RegressionMetrics.mae(y_train, predictions);
    double max_err = RegressionMetrics.max_error(y_train, predictions);
    double r2 = RegressionMetrics.r2(y_train, predictions);
    stdout.printf("Function Approximation Metrics:\nMSE: %.6f, MAE: %.6f, Max Error: %.6f, R2: %.4f\n", mse, mae, max_err, r2);
    for (int i = 0; i < n_samples; i++) {
        double prediction = predictions[i];
        double expected = y_train[i];
        assert(Math.fabs(prediction - expected) < tolerance);
    }
}

int main (string[] args) {
    Test.init(ref args);

    Test.add_func("/multilayer_perceptron/constructor_initialization", test_constructor_initialization);
    Test.add_func("/multilayer_perceptron/forward_propagation", test_forward_propagation);
    Test.add_func("/multilayer_perceptron/backpropagation_xor", test_backpropagation_xor);
    Test.add_func("/multilayer_perceptron/backpropagation_binary_classification", test_backpropagation_binary_classification);
    Test.add_func("/multilayer_perceptron/linear_regression", test_linear_regression);
    Test.add_func("/multilayer_perceptron/function_approximation_sin", test_function_approximation_sin);

    return Test.run();
}