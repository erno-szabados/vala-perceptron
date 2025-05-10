namespace org.esgdev.verceptron {
    public class LayerDefinition : Object {
        public int num_neurons { get; construct; }
        public ActivationFunction activation_function { get; construct; }

        // Constructor for convenience
        public LayerDefinition(int num_neurons, ActivationFunction activation_function) {
            Object(num_neurons: num_neurons, activation_function: activation_function);
        }
    }
}