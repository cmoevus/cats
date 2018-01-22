# -*- coding: utf8 -*-
"""Trackpy interface for CATS tracking."""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp


def filter_features(p, maxmass=None, max_dist_from_center=None, min_dist_from_edges=2):
    """Filter out spurious features.

    Parameters:
    ----------
    p: cats.particles.Particles, pandas.DataFrame
        the dataframe of the detected features
    maxmass: None or number
        the maximum mass of a detection. None for no maximum.
    max_dist_from_center: None or number
        the maximum distance of the center of the particle from the vertical center of the ROI.
    min_dist_from_edges: None or number
        the minimum distance of the center of the particle from the vertical edges of the ROI.

    Returns:
    --------
    particles: same as input
        The filtered particles

    """
    if maxmass is not None:
        p = p[p['mass'] <= maxmass]
    if max_dist_from_center is not None:
        h = p.source.shape[-2]
        hh = int(np.floor(h / 2))
        p = p[np.logical_and(p['y'] >= hh - max_dist_from_center,
                             p['y'] <= hh + max_dist_from_center)]
    if min_dist_from_edges is not None:
        w = p.source.shape[-1]
        p = p[np.logical_and(p['x'] >= min_dist_from_edges,
                             p['x'] <= w - min_dist_from_edges)]
    return p


def filter_particles(p, min_features=3, min_frame_ratio=0.51):
    """Filter out the spurious particles.

    Parameters:
    ----------
    p: pandas.DataFrame, cats.particles.Particles
        the channel(s) to use
    min_features: None or int
        the minimum number of features to be considered a particle.
    min_frame_ratio: None or float in range [0, 1]
        the minimum number of features per frame. For example, if 1/3, there must be at least 10 features detected in a trajectory that goes over 30 frames. This will work in a per feature manner and remove trailing features that are sparse and have a density lower than specified value.

    Returns:
    --------
    particles: the list of particles for each existing channel.

    """
    drop = set()
    for i, particle in p.groupby('particle'):
        if min_frame_ratio is not None:
            # Form a matrix of frame ratios, removing features one by one, starting from the beginning (fw) and from the end (bw)
            frames = particle.frame
            dframe = frames.values - frames.values[:, None]
            feature_pos = np.arange(len(particle))
            dpos = feature_pos - feature_pos[:, None]
            fw_frame_ratio = np.ma.masked_array(dpos, mask=np.triu(dpos, 1) == 0) / np.ma.masked_array(dframe, mask=np.triu(dframe, 1) == 0)
            bw_frame_ratio = np.ma.masked_array(dpos[::-1], mask=np.triu(dpos, 1) == 0) / np.ma.masked_array(dframe[::-1], mask=np.triu(dframe, 1) == 0)

            # Figure out first and last features of the real particle
            fw = fw_frame_ratio >= min_frame_ratio
            fw_score = np.sum(fw, axis=1) - np.sum(np.logical_not(fw), axis=1)
            first_feature = np.argmax(fw_score)
            bw = bw_frame_ratio >= min_frame_ratio
            bw_score = np.sum(bw, axis=1) - np.sum(np.logical_not(bw), axis=1)
            last_feature = np.argmax(bw_score) * -1
            last_feature = last_feature if last_feature != 0 else None

            # Add features to drop list
            drop = drop.union(particle.index[:first_feature])
            if last_feature is not None:
                drop = drop.union(particle.index[last_feature:])

            # Update the particle for subsequent treatment
            particle = particle.iloc[first_feature: last_feature]

        if min_features is not None and len(particle) < min_features:
            drop = drop.union(particle.index)

    return p.drop(drop, axis=0)


def filter_particles_old(p, min_features=3, min_frame_ratio=1 / 3):
    """Filter out the spurious particles.

    Parameters:
    ----------
    p: pandas.DataFrame, cats.particles.Particles
        the channel(s) to use
    min_features: None or int
        the minimum number of features to be considered a particle.
    min_frame_ratio: None or float in range [0, 1]
        the minimum number of features per frame. For example, if 1/3, there must be at least 10 features detected in a trajectory that goes over 30 frames.

    Returns:
    --------
    particles: the list of particles for each existing channel.

    """
    drop = set()
    for i, particle in p.groupby('particle'):
        if min_features is not None and len(particle) < min_features:
            drop.add(i)
        if min_frame_ratio is not None:
            length = particle['frame'].max() + 1 - particle['frame'].min()
            if len(particle) / length < min_frame_ratio:
                drop.add(i)
    for d in drop:
        p = p[p['particle'] != d]
    return p


def merge_particles(p, search_range=3, memory=5):
    """Merge overlapping or proximal particles."""
    # Get the beginning and end of each track
    beginning, end = list(), list()
    for i, particle in p.groupby('particle'):
        beginning.extend(list(particle.index[:memory]))
        end.extend(list(particle.index[-1 * memory - 1:]))
    beginning, end = p.loc[beginning], p.loc[end]

    # Measure the distance between each beginning and each end
    distances = sp.spatial.distance.cdist(beginning[['x', 'y']], end[['x', 'y']])
    frame_distances = np.abs(end['frame'].values - beginning['frame'].values[:, None])

    # Filter out the improper pairs
    within_memory = frame_distances < memory
    different_particle = end['particle'].values != beginning['particle'].values[:, None]
    within_search_range = distances <= search_range
    mask = np.logical_and(within_search_range, np.logical_and(within_memory, different_particle))
    pairs = np.array([(beginning.iloc[i]['particle'], end.iloc[j]['particle'], distances[i][j], frame_distances[i][j]) for i, j in zip(*np.where(mask))])

    # Assemble tracks starting with the closest pairs
    particles = p.copy()
    claimed_beginnings, claimed_ends = list(), list()
    for i, j, d, f in sorted(pairs, key=lambda x: (x[2], x[3])):
        if i not in claimed_beginnings and j not in claimed_ends:
            particles.loc[particles['particle'] == j, 'particle'] = i
            claimed_beginnings.append(i)
            claimed_ends.append(j)

    return particles
