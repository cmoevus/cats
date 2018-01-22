"""Filtering methods for CATS.

This submodule incorporates the different methods for filtering particles and features in CATS:
 - trackpy: based on the trackpy library

The filtering library to use can be selected automatically by setting the `filtering_library` variable in the options.py file to the name of the package. By default, `filtering_library = 'trackpy'`.
"""

from __future__ import absolute_import, division, print_function
import importlib
import cats.options


# Import the default detection library
filtering_library = importlib.import_module('cats.filter.' + cats.options.filtering_library)
features = filtering_library.filter_features
particles = filtering_library.filter_particles
