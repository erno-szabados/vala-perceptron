using GLib; // For Random

public class Perceptron {
    private double[] weights;
    private double bias;
    private double learning_rate = 0.1; // Default learning rate

    // Constructor
    public Perceptron (int num_inputs) {
        // Initialize weights randomly between -1 and 1
        weights = new double[num_inputs];
        var random = new Rand ();
        for (int i = 0; i < num_inputs; i++) {
            // GLib.Random.nextDouble() gives [0, 1), adjust to [-1, 1)
            weights[i] = (Random.next_double () * 2.0) - 1.0;
        }
        // Initialize bias (can also be random or zero)
        bias = (random.next_double () * 2.0) - 1.0;
    }

    // Activation function (Heaviside step function)
    private int activate (double weighted_sum) {
        return (weighted_sum >= 0) ? 1 : 0;
    }

    // Prediction method
    public int predict (double[] inputs) {
        // Ensure input size matches weight size
        assert (inputs.length == weights.length);

        double weighted_sum = 0.0;
        for (int i = 0; i < weights.length; i++) {
            weighted_sum += weights[i] * inputs[i];
        }
        weighted_sum += bias;

        return activate (weighted_sum);
    }

    // Training method (Perceptron Learning Rule)
    public void train (double[] inputs, int target) {
        int prediction = predict (inputs);
        int error = target - prediction; // Error is either 0, 1, or -1

        // Update weights and bias if there's an error
        if (error != 0) {
            for (int i = 0; i < weights.length; i++) {
                weights[i] += learning_rate * error * inputs[i];
            }
            // Update bias (treat it as a weight for a constant input of 1)
            bias += learning_rate * error;
        }
    }

    // Optional: Getter for learning rate
    public double get_learning_rate () { return learning_rate; }

    // Optional: Setter for learning rate
    public void set_learning_rate (double rate) { learning_rate = rate; }
}