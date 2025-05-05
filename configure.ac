AC_PREREQ([2.69])
AC_INIT([verceptron], [0.1.0], [erno.szabados@windowslive.com], [verceptron], [https://example.com/verceptron])
AC_CONFIG_SRCDIR([src/verceptron.vala])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

# Initialize Automake
AM_INIT_AUTOMAKE([foreign -Wall -Werror])

# Set up silent rules (optional, makes build output cleaner)
AM_SILENT_RULES([yes])

# Check for programs
AC_PROG_CC

# Check for Vala compiler
AM_PROG_VALAC([>= 0.56])

# Check for required libraries (GLib/GObject is usually needed by Vala)
PKG_CHECK_MODULES([GLIB], [glib-2.0 >= 2.56 gobject-2.0 >= 2.56])

AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT