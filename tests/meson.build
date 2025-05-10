test_multilayer_perceptron_sources = files(
  'org/esgdev/verceptron/test_multilayer_perceptron.vala',
)

test_error_function_sources = files( 
  'org/esgdev/verceptron/test_error_functions.vala',
)

test_activation_function_sources = files(
  'org/esgdev/verceptron/test_activation_functions.vala',
)

# Define the test executable
test_multilayer_perceptron_executable = executable(
  'test_mlp',
  sources: test_multilayer_perceptron_sources,
  dependencies: [
    glib_dep,
    gobject_dep,
    math_dep,
    verceptron_lib_dep
  ],
  install: false
)

# Define the test executable
test_error_function_executable = executable(
  'test_error_functions',
  sources: test_error_function_sources,
  dependencies: [
    glib_dep,
    gobject_dep,
    math_dep,
    verceptron_lib_dep
  ],
  install: false
)

# Define the test executable
test_activation_function_executable = executable(
  'test_activation_functions',
  sources: test_activation_function_sources,
  dependencies: [
    glib_dep,
    gobject_dep,
    math_dep,
    verceptron_lib_dep
  ],
  install: false
)

# Add individual test cases to Meson's test suite, assigning them to suites
test('mlp_constructor_initialization', test_multilayer_perceptron_executable,
  args : ['-p', '/multilayer_perceptron/constructor_initialization'],
  suite: 'core'
)

test('mlp_forward_propagation', test_multilayer_perceptron_executable,
  args : ['-p', '/multilayer_perceptron/forward_propagation'],
  suite: 'core'
)

test('mlp_backpropagation_xor', test_multilayer_perceptron_executable,
  args : ['-p', '/multilayer_perceptron/backpropagation_xor'],
  suite: 'learning'
)

test('mlp_backpropagation_binary_classification', test_multilayer_perceptron_executable,
  args : ['-p', '/multilayer_perceptron/backpropagation_binary_classification'],
  suite: 'learning'
)
test('test_autoencoder_compression_decompression', test_multilayer_perceptron_executable,
  args : ['-p', '/multilayer_perceptron/autoencoder_compression_decompression'],
  suite: 'learning'
)

test('mlp_linear_regression', test_multilayer_perceptron_executable,
  args : ['-p', '/multilayer_perceptron/linear_regression'],
  suite: 'regression'
)

test('mlp_function_approximation_sin', test_multilayer_perceptron_executable,
  args : ['-p', '/multilayer_perceptron/function_approximation_sin'],
  suite: 'regression'
)

test('error_functions_mean_squared_error', test_error_function_executable,
  args : ['-p', '/error_functions/mean_squared_error'],
  suite: 'error_functions'
)

test('error_functions_binary_cross_entropy', test_error_function_executable,
  args : ['-p', '/error_functions/binary_cross_entropy'],
  suite: 'error_functions'
)

test('activation_functions_identity', test_activation_function_executable,
  args : ['-p', '/activation_functions/identity'],
  suite: 'activation_functions'
)
test('activation_functions_sigmoid', test_activation_function_executable,
  args : ['-p', '/activation_functions/sigmoid'],
  suite: 'activation_functions'
)
test('activation_functions_relu', test_activation_function_executable,
  args : ['-p', '/activation_functions/relu'],
  suite: 'activation_functions'
)
test('activation_functions_leaky_relu', test_activation_function_executable,
  args : ['-p', '/activation_functions/leaky_relu'],
  suite: 'activation_functions'
)
test('activation_functions_tanh', test_activation_function_executable,
  args : ['-p', '/activation_functions/tanh'],
  suite: 'activation_functions'
)