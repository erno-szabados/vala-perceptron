using GLib;
using org.esgdev.verceptron;

// Test MeanSquaredError: checks typical and zero-error cases for loss and gradient
void test_mean_squared_error() {
    var mse = new MeanSquaredError();
    // Typical case
    double target = 1.0;
    double output = 0.8;
    double expected_loss = 0.5 * Math.pow(target - output, 2);
    double expected_grad = output - target;
    assert(Math.fabs(mse.compute(target, output) - expected_loss) < 1e-8);
    assert(Math.fabs(mse.backwards(target, output) - expected_grad) < 1e-8);
    // Zero error: output equals target
    assert(mse.compute(0.5, 0.5) == 0.0);
    assert(mse.backwards(0.5, 0.5) == 0.0);
}

// Test BinaryCrossEntropy: checks typical, edge, and clamped output cases for loss and gradient
void test_binary_cross_entropy() {
    var bce = new BinaryCrossEntropy();
    // Typical case
    double target = 1.0;
    double output = 0.8;
    double eps = 1e-12;
    double clamped_output = Math.fmax(eps, Math.fmin(1.0 - eps, output));
    double expected_loss = -(target * Math.log(clamped_output) + (1.0 - target) * Math.log(1.0 - clamped_output));
    double expected_grad = (clamped_output - target) / (clamped_output * (1.0 - clamped_output));
    assert(Math.fabs(bce.compute(target, output) - expected_loss) < 1e-8);
    assert(Math.fabs(bce.backwards(target, output) - expected_grad) < 1e-8);
    // Edge cases: output exactly 0 or 1, should not be inf or nan
    assert(bce.compute(0.0, 0.0) < 1e-6);
    assert(bce.compute(1.0, 1.0) < 1e-6);
}

int main(string[] args) {
    Test.init(ref args);
    Test.add_func("/error_functions/mean_squared_error", test_mean_squared_error);
    Test.add_func("/error_functions/binary_cross_entropy", test_binary_cross_entropy);
    return Test.run();
}
