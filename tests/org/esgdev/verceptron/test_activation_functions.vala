using GLib;
using org.esgdev.verceptron;

// Test IdentityActivation: should return input for activate, and pass through gradient for backward
void test_identity_activation() {
    var act = new IdentityActivation();
    double z = 2.5;
    double dL_da = 3.0;
    assert(act.activate(z) == z);
    assert(act.backward(z, dL_da) == dL_da);
}

// Test SigmoidActivation: checks output and gradient at z=0
void test_sigmoid_activation() {
    var act = new SigmoidActivation();
    double z = 0.0;
    double dL_da = 1.0;
    double expected = 1.0 / (1.0 + Math.exp(-z));
    assert(Math.fabs(act.activate(z) - expected) < 1e-8);
    double grad = expected * (1.0 - expected);
    assert(Math.fabs(act.backward(z, dL_da) - grad) < 1e-8);
}

// Test ReLUActivation: positive and negative input for activate and backward
void test_relu_activation() {
    var act = new ReLUActivation();
    assert(act.activate(2.0) == 2.0); // positive input
    assert(act.activate(-1.0) == 0.0); // negative input
    assert(act.backward(2.0, 5.0) == 5.0); // positive input
    assert(act.backward(-1.0, 5.0) == 0.0); // negative input
}

// Test LeakyReLUActivation: positive and negative input for activate and backward
void test_leaky_relu_activation() {
    var act = new LeakyReLUActivation();
    double alpha = 0.01;
    assert(act.activate(3.0) == 3.0); // positive input
    assert(Math.fabs(act.activate(-2.0) - (alpha * -2.0)) < 1e-8); // negative input
    assert(act.backward(3.0, 4.0) == 4.0); // positive input
    assert(Math.fabs(act.backward(-2.0, 4.0) - (4.0 * alpha)) < 1e-8); // negative input
}

// Test TanhActivation: checks output and gradient at z=1.0
void test_tanh_activation() {
    var act = new TanhActivation();
    double z = 1.0;
    double dL_da = 2.0;
    double expected = Math.tanh(z);
    assert(Math.fabs(act.activate(z) - expected) < 1e-8);
    double grad = 1.0 - expected * expected;
    assert(Math.fabs(act.backward(z, dL_da) - (dL_da * grad)) < 1e-8);
}

int main(string[] args) {
    Test.init(ref args);
    Test.add_func("/activation_functions/identity", test_identity_activation);
    Test.add_func("/activation_functions/sigmoid", test_sigmoid_activation);
    Test.add_func("/activation_functions/relu", test_relu_activation);
    Test.add_func("/activation_functions/leaky_relu", test_leaky_relu_activation);
    Test.add_func("/activation_functions/tanh", test_tanh_activation);
    return Test.run();
}
