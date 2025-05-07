using GLib;

namespace org.esgdev.verceptron {
    public interface ActivationFunction : Object {
        public abstract double activate(double x);
        public abstract double derivative(double x);
    }

    public class IdentityActivation : Object, ActivationFunction {
        public double activate(double x) {
            return x;
        }

        public double derivative(double x) {
            return 1.0;
        }
    }

    public class SigmoidActivation :  Object, ActivationFunction {
        public double activate(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        public double derivative(double x) {
            double sig_x = activate(x);
            return sig_x * (1.0 - sig_x);
        }
    }

    public class ReLUActivation :  Object, ActivationFunction {
        public double activate(double x) {
            return (x > 0) ? x : 0;
        }

        public double derivative(double x) {
            return (x > 0) ? 1.0 : 0.0;
        }
    }

    public class LeakyRELUActivation :  Object, ActivationFunction {
        private const double ALPHA = 0.01;
        public double activate(double x) {
            return (x > 0) ? x : ALPHA * x;
        }

        public double derivative(double x) {
            return (x > 0) ? 1.0 : ALPHA;
        }
    }
}

