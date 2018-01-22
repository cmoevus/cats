# -*- coding: utf8 -*-
"""Classes and functions for deal with DNA molecules and their content."""
from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
from glob import glob
import itertools
import skimage.io
import os

import cats.images
from cats import extensions
import cats.detect


@extensions.append
class DNAs(list):
    """Group DNA molecules together."""

    def __init__(self, *args, **kwargs):
        """Instanciate the object."""
        super().__init__(*args, **kwargs)

    def draw_particles(self, folder):
        """Draw all particles for each DNA molecules and save it in the given folder."""
        for i, dna in enumerate(self):
            skimage.io.imsave(os.path.join(folder, '{}.tif').format(i), dna.draw_particles())


@extensions.append
class DNA(object):
    """Useful for dna molecules.

    Parameters:
    -----------
    dataset: cat.Images object
        the dataset to extract the dna from
    point: 2-tuple of 2-tuples of ints
        the positions in (x, y) of the start and end points, respectively, of the dna molecule in the dataset.
    name: string
        name of the content in the channel (optional)

    One can list as many (dataset, pt1, pt2, name) elements as they have channels, by putting them between parantheses, with an optional name to the channel

    """

    def __init__(self, *channels):
        """Set up the DNA molecule."""
        if type(channels[0]) is cats.Images:  # Single channel
            channels = [channels]
        self.datasets, self.points, self.names = [], [], []
        for i, c in enumerate(channels):
            self.datasets.append(c[0])
            self.points.append(((int(round(c[1][0][0], 0)), int(round(c[1][0][1], 0))),
                               (int(round(c[1][1][0], 0)), int(round(c[1][1][1], 0)))))
            if len(c) > 2:
                self.names.append(c[2])
            else:
                self.names.append(None)

    def kymogram(self):
        """Extract the kymogram of the dna molecule from the dataset."""
        if not hasattr(self, '_kymo'):
            self._kymo = list()
            for c in range(len(self.datasets)):
                self._kymo.append(cats.Kymogram.from_coordinates(*self.points[c], self.datasets[c]))
        return self._kymo

    @kymogram.setter
    def kymogram(self, k):
        self._kymo = k

    @property
    def roi(self):
        """Extract the region of interest around the location of DNA molecule."""
        #
        # Note: this sucks. I should rather use skimage.draw.polygon to draw the ROI. However, the issue is that cats.Images doesn't support advanced indexing. I should first implement advanced indexing.
        #

        # Generate the Region Of Interest from the dna
        if not hasattr(self, '_roi'):
            # Define the section size
            if not hasattr(self, 'roi_half_height'):
                self.roi_half_height = 3

            # Build the ROI object
            self._roi = list()
            for i, dataset in enumerate(self.datasets):
                x0, y0 = self.points[i][0]
                x1, y1 = self.points[i][1]
                y_max, x_max = dataset.shape
                # Consider the orientation of the data
                x_direction = -1 if x0 > x1 else 1
                if y0 > y1:
                    y_slice = slice(min(y_max, y0 + self.roi_half_height), max(0, y1 - self.roi_half_height), -1)
                else:
                    y_slice = slice(max(0, y0 - self.roi_half_height), min(y_max, y1 + self.roi_half_height), 1)
                x_slice = slice(max(0, x0), min(x1, x_max), x_direction)
                roi = dataset[:, y_slice, x_slice]
                roi.parent_slices = (x_slice, y_slice)  # Info on where this ROI comes from, in x and y
                self._roi.append(roi)
        return self._roi

    @roi.setter
    def roi(self, roi):
        self._roi = roi

    def roi_slice(self):
        """Return the coordinates of the ROI slice, as (x slice, y slice)."""

    def track(self, channels=False, **kwargs):
        """Track particles from the DNA's region of interest.

        Parameters:
        -----------
        channels: int, list of ints
            the channel(s) to track
        any keyword argument to be passed to the `cats.detect.track` function.

        """
        # Setup
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]
        if not hasattr(self, 'particles'):
            self.particles = list(None for d in self.datasets)

        for channel in channels:
            self.particles[channel] = cats.detect.track(self.roi[channel], **kwargs)

        return [self.particles[channel] for channel in channels] if len(channels) > 1 else self.particles[channels[0]]

    def draw_particles(self, channels=False):
        """Draw each detection of each particle onto the kymogram."""
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]

        kymos = list()
        for channel in channels:
            kymo = self.kymogram.draw(channels=channel)
            for i, particle in self.particles[channel].groupby('particle'):
                color = cats.colors.random()
                for i, feature in particle.iterrows():
                    kymo[int(round(feature['x'], 0)), int(feature['frame'])] = color
            kymos.append(kymo)

        return kymos[0] if len(kymos) < 2 else cats.images.stack_images(*kymos)


def populate_kymograms(dnas):
    """Populate the kymograms for a list of DNA molecules that share common datasets.

    Save time on opening images by populating kymograms for several DNA molecules every time an image is opened.
    """
    # Build empty kymograms
    for dna in dnas:
        dna.kymogram.build_empty()

    # Populate kymograms per dataset
    per_datasets = itertools.groupby([(dna, channel, dataset) for dna in dnas for channel, dataset in enumerate(dna.datasets)], key=lambda x: x[2])
    for dataset, dataset_dnas in per_datasets:
        dataset_dnas = list(dataset_dnas)
        for f, frame in enumerate(dataset):
            for dna, channel, d in dataset_dnas:
                if f < dna.kymogram.shape[2]:  # Do not try to exceed the kymogram's length
                    dna.kymogram.raw[channel, :, f] = frame[dna.kymogram.pixel_positions[channel]]

    # Copy kymogram to main kymogram
    for dna in dnas:
        dna.kymogram.reset()


def csv_to_coordinates(csv_file):
    """Transform a CSV lines list from ImageJ into a list of dna coordinates.

    How to use:
    -----------
    - Draw a line over the DNA
    - Measure it (Ctrl+M)
    - Make sure you have the following measurements set:
        - Boundary rectangle
    - Save with the column headers on.
    """
    lines = pd.read_csv(csv_file)
    lines['EX'] = lines['BX'] + lines['Width']
    lines['EY'] = lines['BY'] - lines['Height'] * lines['Angle'] / abs(lines['Angle'])
    return [((r[1]['BX'], r[1]['BY']), (r[1]['EX'], r[1]['EY'])) for r in lines.iterrows()]


def generate_registration_function(reference, relative):
    """Generate a registration function that transforms coordinates (x, y) from a reference source into (x, y) to the relative source.

    Parameters:
    -----------
    reference: 2-tuple
        the coordinates (x0, y0) of the reference point
    relative: 2-tuple
        the coordinates (x1, y1) of the relative point

    Returns:
    --------
    register: func
        a function that transforms reference points into relative points

    """
    dx = relative[0] - reference[0]
    dy = relative[1] - reference[1]

    def register(x, y):
        """Transform coordinates (x,y) of the reference into (x,y) of the relative dataset."""
        return x + dx, y + dy

    return register


def invert_coordinates(coords):
    """Invert the orientation of a list of kymogram coordinates, so that the top of the kymogram becomes the bottom."""
    return [(pt2, pt1) for pt1, pt2 in coords]


def register_coordinates(reg_func, coords):
    """Register a list of coordinates as pt1, pt2."""
    coords = np.array(coords.copy())
    coords[:, 0, 0], coords[:, 0, 1] = reg_func(coords[:, 0, 0], coords[:, 0, 1])
    coords[:, 1, 0], coords[:, 1, 1] = reg_func(coords[:, 1, 0], coords[:, 1, 1])
    return coords


def hand_picked_files_to_df(path):
    """Transform your loosy .csv files from ImageJ into a magnificient pandas DataFrame."""
    particles = list()
    for i, f in enumerate(glob(path)):
        values = pd.read_csv(f, header=None)
        values.rename(columns={0: 'x', 1: 'y'}, inplace=True)
        values['id'] = i
        particles.append(values)

    return pd.concat(particles) if len(particles) > 0 else pd.DataFrame([], columns=['x', 'y', 'id'])
