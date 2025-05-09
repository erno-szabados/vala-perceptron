# Create a static library for the verceptron core functionality
# This will allow the test suite to use it
verceptron_lib = static_library(
  'verceptron-lib',
  sources: files(
    'org/esgdev/verceptron/perceptron.vala',
    'org/esgdev/verceptron/multilayer_perceptron.vala',
    'org/esgdev/verceptron/activation_functions.vala',
    'org/esgdev/verceptron/error_functions.vala',
    'org/esgdev/verceptron/layer_definition.vala',
  ),
  dependencies: [glib_dep, gobject_dep, math_dep],  # Added math_dep
)

# Create the verceptron_lib dependency for tests to use
verceptron_lib_dep = declare_dependency(
  link_with: verceptron_lib,
  include_directories: include_directories('.')
)
