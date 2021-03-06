# -*- coding: utf8 -*-
"""Trackpy interface for CATS tracking."""
from __future__ import absolute_import, division, print_function
import trackpy as tp
import numpy as np
import itertools
import cats.particles
from cats.filter.trackpy import filter_particles, filter_features, merge_particles


def track_features(images, diameter=5, **kwargs):
    """Locate features - call them blobs, detections - in the given images.

    This is a wrapper for the `trackpy.batch` function.
    See its documentation for parameters and more information.

    Parameters:
    ----------
    diameter: odd int
        the approximate diameter, in pixels, of the features to detect.
    any keyword argument to be passed to the trackpy ``batch`` function

    Returns:
    --------
    features: pd.DataFrame
        the list of detected features

    """
    df = tp.batch(images, diameter=diameter, **kwargs)
    kwargs['diameter'] = diameter
    features = cats.particles.Particles(df)

    # Set up element attributes to be added once the particles are tracked
    features._element_attributes = {'source': images,
                                    'tracking_parameters': kwargs}

    return features


def track_particles(df, search_range=3, memory=5, **kwargs):
    """Assemble particles - call them tracks, trajectories - from the features located in the DNA's region of interest.

    This is a wrapper for the `trackpy.link_df` function.
    See its documentation for parameters and more information.

    Parameters:
    ----------
    search_range: int
        The radius of pixels around a feature in which to search for the next feature.
    memory: int
        The max number of frames between two features for them to be considered the same particle
    any keyword argument to be passed to the trackpy ``link_df`` function.

    Returns:
    --------
    particles: cats.particles.Particles
        The Particles found

    """
    ps = cats.particles.Particles(tp.link_df(df, search_range=search_range, memory=memory, **kwargs))

    # Setup element attributes from Features
    if hasattr(df, '_element_attributes') and 'particle' not in df.columns:
        for attr, value in df._element_attributes.items():
            ps._set_element_attribute(attr, value)
    kwargs['search_range'] = search_range
    kwargs['memory'] = memory
    ps._update_element_attribute('tracking_parameters', kwargs, ps)
    return ps


def track(images, features=True, particles=True, filter_feats=True, filter_parts=True, merge_parts=True, **kwargs):
    """Track particles from images, wrapper function.

    First calls `trackpy.batch` to get features, then filters the result using `cats.detect.trackpy.filter_features`.
    Then calls `trackpy.link_df` to get particles, then filters the result using `cats.detect.trackpy.filter_particles`.

    Parameters:
    ----------
    images: cats.Images object
        the images to track from
    features: bool
        Whether to track the features in the given images. If false, a pandas.DataFrame of features must be given, instead of images.
    particles: bool
        Whether to link features into particles.
    filter_feats: bool
        Whether to apply filters on detected features to clean them up.
    filter_parts: bool
        Whether to apply filters on detected particles to clean them up.
    merge_parts: bool
        Whether to merge adjacent particles that are withing `memory` frames and `search_range` pixels after clean up.

    See `trackpy.batch`, `trackpy.link_df`, `cats.filter.trackpy.filter_features`, `cats.filter.filter_particles` for necessary tracking and filtering parameters.

    """
    # Split the kwargs appropriately
    feat_kwargs = dict((k, v) for k, v in kwargs.items() if k in tp.batch.__code__.co_varnames)
    filt_feat_kwargs = dict((k, v) for k, v in kwargs.items() if k in filter_features.__code__.co_varnames)
    filt_part_kwargs = dict((k, v) for k, v in kwargs.items() if k in filter_particles.__code__.co_varnames)
    part_kwargs = dict((k, v) for k, v in kwargs.items() if k in tp.link_df.__code__.co_varnames)
    merge_kwargs = dict((k, v) for k, v in kwargs.items() if k in merge_particles.__code__.co_varnames)

    # Do the job
    if features:
        features = track_features(images, **feat_kwargs)
        if filter_feats:
            features = filter_features(features, **filt_feat_kwargs)
    else:
        features = images

    if particles:
        particles = track_particles(features, **part_kwargs)
        if filter_parts:
            particles = filter_particles(particles, **filt_part_kwargs)
        if merge_parts:
            particles = merge_particles(particles, **merge_kwargs)

        particles.renumber_particles(copy=False)  # Clean up the mess

    else:
        particles = features

    return particles


def estimate_tracking_parameters(images, diameter=range(3, 8, 2), smoothing_size=range(1, 5, 2), noise_size=np.arange(0.6, 0.99, 0.02)):
    """Estimate the tracking parameters for the given images.

    Parameters:
    -----------
    images: cats.Images
        The images to track from. Usually, a subset is sufficient (about a few hundred features).
    diameter: list
        Potential values for the `diameter` parameter of the `trackpy.locate` function.
    smoothing_size: list
        Potential values for the `smoothing_size` parameter of the `trackpy.locate` function.
    noise_size: list
        Potential values for the `noise_size` parameter of the `trackpy.locate` function.

    """
    scores = list()
    for i, (d, ss, ns) in enumerate(itertools.product(diameter, smoothing_size, noise_size)):
        feats = cats.detect.features(images, diameter=d, smoothing_size=ss, noise_size=ns)
        score = np.mean([np.var(np.histogram([x - np.int(x) for x in feats['y']])[0]), np.var(np.histogram([x - np.int(x) for x in feats['x']])[0])])
        if score > 0:
            scores.append((('diameter', d), ('smoothing_size', ss), ('noise_size', ns), score))
    return dict(sorted(scores, key=lambda x: x[3])[0][:3])
