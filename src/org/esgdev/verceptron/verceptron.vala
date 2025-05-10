namespace org.esgdev.verceptron {

    public static int main (string[] args) {
        stdout.printf ("Verceptron: Training an AND gate...\n");

        // 1. Define training data for AND gate
        // Inputs: [0,0], [0,1], [1,0], [1,1]
        // Targets: 0,    0,    0,    1
        double[,] training_inputs = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        };
        int[] training_targets = {0, 0, 0, 1};

        // 2. Instantiate the Perceptron
        // AND gate has 2 inputs
        var p = new Perceptron (2);
        // Optional: Adjust learning rate if needed
        // p.set_learning_rate(0.05);

        // 3. Train the Perceptron
        int epochs = 100; // Number of times to iterate through the training data
        stdout.printf ("Training for %d epochs...\n", epochs);
        // Get the length of the first dimension for iteration
        for (int i = 0; i < training_inputs.length[0]; i++) {
            // Extract the i-th row as a single-dimensional array for the train method
            // This requires creating a temporary slice/copy.
            double[] current_input = { training_inputs[i, 0], training_inputs[i, 1] };
            p.train (current_input, training_targets[i]);
        }
        stdout.printf ("Training complete.\n");

        // 4. Run inference (predict)
        stdout.printf ("Testing the trained perceptron:\n");
        // Get the length of the first dimension for iteration
        for (int i = 0; i < training_inputs.length[0]; i++) {
            // Extract the i-th row for the predict method
            double[] current_input = { training_inputs[i, 0], training_inputs[i, 1] };
            int prediction = p.predict (current_input);
            stdout.printf ("Input: [%.1f, %.1f] -> Target: %d, Prediction: %d\n",
                training_inputs[i, 0], training_inputs[i, 1], training_targets[i], prediction);
        }

        return 0;
    }
}
