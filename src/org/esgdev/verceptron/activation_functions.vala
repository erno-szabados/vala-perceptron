using GLib;

namespace org.esgdev.verceptron {
    public interface ActivationFunction : Object {
        public abstract double activate(double z);
        // backward takes z (the input to the activation function) and dL_da (the gradient of loss w.r.t. activation output 'a')
        // it returns dL_dz = dL_da * (da/dz)
        public abstract double backward(double z, double dL_da);
    }

    public class IdentityActivation : Object, ActivationFunction {
        public double activate(double z) {
            return z;
        }

        public double backward(double z, double dL_da) {
            // da/dz = 1.0
            return dL_da * 1.0;
        }
    }

    public class SigmoidActivation :  Object, ActivationFunction {
        public double activate(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }

        public double backward(double z, double dL_da) {
            double sig_z = activate(z); // Or: 1.0 / (1.0 + Math.exp(-z));
            double da_dz = sig_z * (1.0 - sig_z);
            return dL_da * da_dz;
        }
    }

    public class ReLUActivation :  Object, ActivationFunction {
        public double activate(double z) {
            return (z > 0) ? z : 0.0;
        }

        public double backward(double z, double dL_da) {
            double da_dz = (z > 0) ? 1.0 : 0.0;
            return dL_da * da_dz;
        }
    }

    public class LeakyReLUActivation :  Object, ActivationFunction {
        private const double ALPHA = 0.01;
        public double activate(double z) {
            return (z > 0) ? z : ALPHA * z;
        }

        public double backward(double z, double dL_da) {
            double da_dz = (z > 0) ? 1.0 : ALPHA;
            return dL_da * da_dz;
        }
    }

    public class TanhActivation :  Object, ActivationFunction {
        public double activate(double z) {
            return Math.tanh(z);
        }

        public double backward(double z, double dL_da) {
            double tanh_z = Math.tanh(z); // Could also use activate(z)
            double da_dz = 1.0 - tanh_z * tanh_z;
            return dL_da * da_dz;
        }
    }
}

