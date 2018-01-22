# -*- coding: utf8 -*-
"""Trackpy interface for CATS tracking."""
from __future__ import absolute_import, division, print_function
import trackpy as tp
import cats.particles
from cats.filter.trackpy import filter_particles, filter_features, merge_particles

def track_features(images, diameter=3, **kwargs):
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
    df.source = None  # Just a stoopid hack to avoid pandas' warning
    df.source = images
    return cats.particles.Particles(df, source=images)


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
    if hasattr(df, 'source'):
        ps.source = df.source
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
    else:
        particles = features

    return particles
