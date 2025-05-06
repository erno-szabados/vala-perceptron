verceptron_sources = files(
  'org/esgdev/verceptron/verceptron.vala',
  'org/esgdev/verceptron/perceptron.vala',
  'org/esgdev/verceptron/multilayer_perceptron.vala',
)

# Create a static library for the verceptron core functionality
# This will allow the test suite to use it
verceptron_lib = static_library(
  'verceptron-lib',
  sources: files(
    'org/esgdev/verceptron/perceptron.vala',
    'org/esgdev/verceptron/multilayer_perceptron.vala',
  ),
  dependencies: [glib_dep, gobject_dep],
)

# Create the verceptron_lib dependency for tests to use
verceptron_lib_dep = declare_dependency(
  link_with: verceptron_lib,
  include_directories: include_directories('.')
)

# Define the executable
executable('verceptron',
  sources: verceptron_sources,
  dependencies: [glib_dep, gobject_dep],
  install: true,
)