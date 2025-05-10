using GLib;

namespace org.esgdev.verceptron {
    public interface ErrorFunction : Object {
        // Compute the error given the target and output
        public abstract double compute(double target, double output);
        // Compute the backward pass (gradient of the error with respect to the output)
        public abstract double backwards(double target, double output);
    }

    // Mean Squared Error (MSE) implementation
    public class MeanSquaredError : Object, ErrorFunction {
        public double compute(double target, double output) {
            return 0.5 * Math.pow(target - output, 2);
        }
        public double backwards(double target, double output) {
            return output - target;
        }
    }

    // Example: Cross-Entropy for binary classification
    public class BinaryCrossEntropy : Object, ErrorFunction {
        public double compute(double target, double output) {
            // Clamp output to avoid log(0)
            double eps = 1e-12;
            output = Math.fmax(eps, Math.fmin(1.0 - eps, output));
            return -(target * Math.log(output) + (1.0 - target) * Math.log(1.0 - output));
        }
        public double backwards(double target, double output) {
            double eps = 1e-12;
            output = Math.fmax(eps, Math.fmin(1.0 - eps, output));
            return (output - target) / (output * (1.0 - output));
        }
    }
}
