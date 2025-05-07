test_sources = files(
  'org/esgdev/verceptron/test_multilayer_perceptron.vala',
)

# Define the test executable
test_executable = executable(
  'test_verceptron',
  sources: test_sources,
  dependencies: [
    glib_dep,
    gobject_dep,
    math_dep,
    verceptron_lib_dep
  ],
  install: false
)

# Add individual test cases to Meson's test suite, assigning them to suites
test('mlp_constructor_initialization', test_executable,
  args : ['-p', '/multilayer_perceptron/constructor_initialization'],
  suite: 'core'
)

test('mlp_forward_propagation', test_executable,
  args : ['-p', '/multilayer_perceptron/forward_propagation'],
  suite: 'core'
)

test('mlp_backpropagation_xor', test_executable,
  args : ['-p', '/multilayer_perceptron/backpropagation_xor'],
  suite: 'learning'
)

test('mlp_backpropagation_binary_classification', test_executable,
  args : ['-p', '/multilayer_perceptron/backpropagation_binary_classification'],
  suite: 'learning'
)