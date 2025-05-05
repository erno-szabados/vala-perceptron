# Define the executable program to be built and installed in bindir
bin_PROGRAMS = verceptron

# List the source files for the program
verceptron_SOURCES = \
       $(srcdir)/verceptron.vala

# Pass flags required by Vala (linking GLib/GObject)
verceptron_VALAFLAGS = --pkg glib-2.0 --pkg gobject-2.0
verceptron_CFLAGS = $(GLIB_CFLAGS)

# Link with libraries found by PKG_CHECK_MODULES in configure.ac
verceptron_LDADD = $(GLIB_LIBS)