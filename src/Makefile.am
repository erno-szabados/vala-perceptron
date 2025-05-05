# Define the executable program to be built and installed in bindir
bin_PROGRAMS = verceptron

# List the source files for the program
verceptron_SOURCES = \
       verceptron.vala \
       perceptron.vala

# Pass flags required by Vala (linking GLib/GObject)
verceptron_VALAFLAGS = --pkg glib-2.0 --pkg gobject-2.0 -d $(builddir)
# Add GLib CFLAGS and suppress the incompatible pointer type warning from g_once_init
AM_CFLAGS = $(GLIB_CFLAGS) -Wno-incompatible-pointer-types

# Link with libraries found by PKG_CHECK_MODULES in configure.ac
verceptron_LDADD = $(GLIB_LIBS)

CLEANFILES = verceptron.c perceptron.c verceptron_vala.stamp verceptron.o perceptron.o verceptron
