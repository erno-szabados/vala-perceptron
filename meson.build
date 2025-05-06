project('verceptron', ['c', 'vala'],
  version : '0.1.0',
  license : 'MIT', 
  meson_version : '>= 0.56.0',
  default_options : [
    'warning_level=2',
    'werror=true'
  ]
)

# Project metadata
project_description = 'Verceptron, an example application'
project_url = 'https://example.com/verceptron'
project_bugreport = 'erno.szabados@windowslive.com'

# Add project-wide C arguments
add_project_arguments('-Wno-incompatible-pointer-types', language : 'c')

# Dependencies
required_glib_version = '>= 2.56'
glib_dep = dependency('glib-2.0', version : required_glib_version, required : true)
gobject_dep = dependency('gobject-2.0', version : required_glib_version, required : true)
math_dep = meson.get_compiler('c').find_library('m', required : true)  # Add math library

# Configuration data for config.h
conf_data = configuration_data()
conf_data.set_quoted('PACKAGE_NAME', meson.project_name())
conf_data.set_quoted('PACKAGE_VERSION', meson.project_version())
conf_data.set_quoted('PACKAGE_STRING', meson.project_name() + ' ' + meson.project_version())
conf_data.set_quoted('PACKAGE_BUGREPORT', project_bugreport)
conf_data.set_quoted('PACKAGE_URL', project_url)

configure_file(
  output : 'config.h',
  configuration : conf_data
)

# Include source and test directories
subdir('src')
subdir('tests')
