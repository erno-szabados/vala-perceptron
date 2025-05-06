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
    math_dep,  # Added math_dep
    verceptron_lib_dep
  ],
  install: false
)

# Add the test to Meson's test suite
test('verceptron_tests', test_executable)