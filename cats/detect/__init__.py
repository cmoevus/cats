"""Detection methods for CATS.

This submodule incorporates the different methods for tracking particles in CATS:
 - trackpy: based on the trackpy library

The detection library to use can be selected automatically by setting the `detection_library` variable in the options.py file to the name of the package. By default, `detection_library = 'trackpy'`.
"""

from __future__ import absolute_import, division, print_function
import importlib
import numpy as np
import scipy as sp
import cats.options


# Import the default detection library
detection_library = importlib.import_module('cats.detect.' + cats.options.detection_library)
features = detection_library.track_features
filter_features = detection_library.filter_features
particles = detection_library.track_particles
filter_particles = detection_library.filter_particles
track = detection_library.track


#
# Other functions
#
def particles_using_handtracks(particles, hand_particles, dist=2, allow_duplication=False):
    """Match handpicked particles with features that are tracked by software.

    Parameters:
    -----------
    particles: pandas.DataFrame or cats.particles.Particles or cats.particles.Particle
    hand_particles: pandas.DataFrame or cats.particles.Particles or cats.particles.Particle as obtained from `cats.particles.from_kymogram_handtracks`
        the manually selected particles (with all frames extrapolated)
    dist: int
        the maximal distance features can be from the points in the tracks or infered from the tracks.
    allow_duplication: bool
        Whether the same feature can be used for two different particles

    Track format:
    -------------
    A track is a list of features, in the format (x, y, f), with x, the position in x, y, the position in y and f, the frame at which the feature is found. Hence, the `tracks` parameter is a list of lists of tuples.
    The features defined by the tracks only have to be those between the segments of linear movement of the particle. The features in between those defined will be inferred (linearly).

    Notes
    ------
    Make sure that the coordinates in the DataFrame are from the same reference as in the tracks. For example, tracking will often return values from an ROI.

    """
    distances = sp.spatial.distance.cdist(hand_particles[['x', 'y']], particles[['x', 'y']])
    less_than_dist = distances <= dist
    same_frame = hand_particles['frame'].values[:, np.newaxis] == particles['frame'].values
    adjusted_distances = np.where(np.logical_and(less_than_dist, same_frame), distances, np.inf)
    matches = np.array([np.arange(0, len(hand_particles)), np.argmin(adjusted_distances, axis=1), np.min(adjusted_distances, axis=1)])
    valid_particles = matches[2] != np.inf
    filtered = particles.iloc[matches[1][valid_particles]].copy()
    filtered['particle'] = hand_particles.iloc[matches[0][valid_particles]]['particle'].values
    if not allow_duplication:
        filtered = filtered[np.logical_not(filtered.duplicated(['x', 'y', 'frame']))].copy()
    return filtered
