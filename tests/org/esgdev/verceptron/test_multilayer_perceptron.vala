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

int main (string[] args) {
    Test.init (ref args);

    // Add test cases
    Test.add_func ("/multilayer_perceptron/constructor_initialization", test_constructor_initialization);
    Test.add_func ("/multilayer_perceptron/forward_propagation", test_forward_propagation);

    return Test.run ();
}