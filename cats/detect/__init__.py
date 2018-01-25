"""Detection methods for CATS.

This submodule incorporates the different methods for tracking particles in CATS:
 - trackpy: based on the trackpy library

The detection library to use can be selected automatically by setting the `detection_library` variable in the options.py file to the name of the package. By default, `detection_library = 'trackpy'`.
"""

from __future__ import absolute_import, division, print_function
import importlib
import cats.options


# Import the default detection library
detection_library = importlib.import_module('cats.detect.' + cats.options.detection_library)
features = detection_library.track_features
filter_features = detection_library.filter_features
particles = detection_library.track_particles
filter_particles = detection_library.filter_particles
track = detection_library.track
