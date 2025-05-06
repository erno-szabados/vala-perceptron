project('verceptron', ['c', 'vala'],
  version : '0.1.0',
  license : 'MIT', 
  meson_version : '>= 0.56.0', # A reasonable minimum Meson version
  default_options : [
    'warning_level=2', # Enables many useful warnings, akin to -Wall
    'werror=true'      # Treat warnings as errors, from AM_INIT_AUTOMAKE
  ]
)

# Project metadata (from AC_INIT)
project_description = 'Verceptron, an example application' # You can expand on this
project_url = 'https://example.com/verceptron'
project_bugreport = 'erno.szabados@windowslive.com'

# Add project-wide C arguments.
# The -Wno-incompatible-pointer-types was in your AM_CFLAGS.
# GLib's CFLAGS are automatically included by the dependency object.
add_project_arguments('-Wno-incompatible-pointer-types', language : 'c')

# Dependencies (from PKG_CHECK_MODULES and verceptron_VALAFLAGS)
# Meson's dependency system handles finding libraries and their VAPI files for Vala,
# as well as CFLAGS and LIBS for the C compiler.
required_glib_version = '>= 2.56'
glib_dep = dependency('glib-2.0', version : required_glib_version, required : true)
gobject_dep = dependency('gobject-2.0', version : required_glib_version, required : true)

# Configuration data for config.h (from AC_CONFIG_HEADERS)
# This replicates some common defines Autotools puts in config.h.
# Meson will generate this file in the build directory, and it will be
# automatically available to C sources (including Vala-generated C code).
conf_data = configuration_data()
conf_data.set_quoted('PACKAGE_NAME', meson.project_name())
conf_data.set_quoted('PACKAGE_VERSION', meson.project_version())
conf_data.set_quoted('PACKAGE_STRING', meson.project_name() + ' ' + meson.project_version())
conf_data.set_quoted('PACKAGE_BUGREPORT', project_bugreport)
conf_data.set_quoted('PACKAGE_URL', project_url)
# conf_data.set('GETTEXT_PACKAGE', meson.project_name()) # Add if you use gettext

configure_file(
  output : 'config.h',
  configuration : conf_data
)

# Source files for the executable (from src/Makefile.am verceptron_SOURCES)
verceptron_sources = [
  'src/verceptron.vala',
  'src/perceptron.vala'
]

# Define the executable (from src/Makefile.am bin_PROGRAMS)
executable('verceptron',
  verceptron_sources,
  dependencies : [glib_dep, gobject_dep],
  install : true # Installs to 'bindir' by default
)