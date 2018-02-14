# -*- coding: utf8 -*-
"""Classes and functions for deal with DNA molecules and their content."""
from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
from glob import glob
import os
import skimage.io
import pims

from . import extensions
import cats.utils
import cats.images
import cats.detect
import cats.kymograms
import cats.colors


@extensions.append
class DNAs(list, cats.utils.pickle_save):
    """Group DNA molecules together.

    Features:
    ---------
    Get DNA molecules:
    >>> dnas[0], dnas[:10]
    Get channels:
    >>> dnas['protein']
    Mass-run DNA methods/properties
    >>> dnas.length
    >>> dnas[:10].track()
    >>>dnas['protein'].track()
    """

    def __init__(self, *args, **kwargs):
        """Instanciate the object."""
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        """Return the queried item and make it a `DNAs` object instead of a list."""
        # User wants channels
        if type(item) is str:
            return DNAs([dna[item] for dna in self])

        # User wants something else
        ret = super().__getitem__(item)
        if type(ret) is list:
            ret = DNAs(ret)
        return ret

    def __getattr__(self, attr):
        """Use methods and attributes in underlying DNA molecules."""
        # Get attribute from the underlying objects
        if hasattr(DNA, attr):
            obj = DNA
        elif hasattr(DNAChannel, attr):
            obj = DNAChannel
        else:
            obj = None
        if obj:
            if type(getattr(obj, attr)) == property:
                return [getattr(dna, attr) for dna in self]
            else:
                def all_attr(*args, **kwargs):
                    return [getattr(dna, attr)(*args, **kwargs) for dna in self]
                return all_attr
        elif any([hasattr(v, attr) for v in self]):
            return [getattr(dna, attr, None) for dna in self]
        return super().__getattribute__(attr)

    @staticmethod
    def from_csv(f, images, name=None, invert_coordinates=False):
        """Import DNA molecules from CSV files.

        Transform a CSV lines list from ImageJ into a list of dna coordinates.

        Parameters:
        ----------
        f: str
            the CSV file
        images: cats.images.Images
            the images in which the DNA molecules are recorded
        name: str
            the name of the channel in the CSV file. If None, will create single-channel objects.
        invert_coordinates: bool
            whether to invert the coordinates in the file, so that the top of the kymogram becomes the bottom. Useful if your images have pedestals at the left and barriers at the right.

        How to use:
        -----------
        - Draw a line over the DNA
        - Measure it (Ctrl+M)
        - Make sure you have the following measurements set:
            - Boundary rectangle
        - Save with the column headers on.

        """
        lines = pd.read_csv(f)
        lines['EX'] = lines['BX'] + lines['Width']
        lines['EY'] = lines['BY'] - lines['Height'] * lines['Angle'] / abs(lines['Angle'])
        coords = [((r[1]['BX'], r[1]['BY']), (r[1]['EX'], r[1]['EY'])) for r in lines.astype(int).iterrows()]
        if invert_coordinates:
            coords = cats.dna.invert_coordinates(coords)
        return DNAs([DNA(name, images, beg, end) if name is not None else DNA(images, beg, end) for beg, end in coords])

    def add_channel(self, name, images, coordinates):
        """Add a channel to all the DNA molecules.

        Parameters:
        ------------
        name: str
            the name of the channel
        images: cats.images.Images
            the images in which the channel is recorded
        coordinates: list of 2-tuples of 2-tuples
            the coordinates of the channel for each DNA molecule, for example as obtained from `DNAs.register_channel`.

        """
        for i, dna in enumerate(self):
            dna.add_channel(name, images, *coordinates[i])

    def register_channel(self, name, images, reg_func, reference=None):
        """Add a channel and get the coordinates based on another channel's coordinates.

        Parameters:
        -----------
        name: str
            the name of the channel
        images: cats.images.Images
            the images in which the channel is recorded
        reg_func: function
            the registration function that takes in the beginning and end points of the reference channel and makes it the coordinates of the new channel.
        reference: str or None
            the name of the reference channel. If None and the DNA molecules only have one channel, will use that one.

        Notes:
        ------
        The utility function `cats.dna.generate_registration_function` can be used to generate `reg_func`.

        """
        if reference is None:
            if len(self) == 1:
                reference = tuple(self.keys())[0]
            else:
                raise ValueError('Cannot determine the reference channel.')
        coordinates = [(reg_func(*dna[reference].beginning), reg_func(*dna[reference].end)) for dna in self]
        self.add_channel(name, images, coordinates)

    def generate_kymograms(self):
        """Group DNA molecules per images and accelerate the kymogram-writing time by opening images only once."""
        # Sort molecules by images
        images = list()
        images_stack = dict()
        for dna in self:
            for dna_chn in dna.values():
                try:
                    key = images.index(dna_chn.images)
                except ValueError:
                    key = len(images)
                    images.append(dna_chn.images)
                    images_stack[key] = list()
                images_stack[key].append(dna_chn)

        # Build kymograms
        for key, dnas in images_stack.items():
            kymos = [[] for i in dnas]
            for image in images[key]:
                for i, dna in enumerate(dnas):
                    kymos[i].append(image[dna.pixels])
            for i, dna in enumerate(dnas):
                dna.kymogram = np.array(kymos[i]).T.view(cats.kymograms.Kymogram)

    def save_kymograms(self, folder):
        """Write kymograms for all DNA molecules in the given folder."""
        for i, dna in enumerate(self):
            if type(dna) is DNAChannel:
                dna.kymogram.save(os.path.join(folder, '{}.tif').format(i))
            else:
                for name, channel in dna.channels:
                    channel.kymogram.save(os.path.join(folder, '{}.{}.tif').format(i, name))

    def draw_particles(self, folder):
        """Draw all particles for each DNA molecules and save it in the given folder."""
        for i, dna in enumerate(self):
            if type(dna) is DNAChannel:
                skimage.io.imsave(os.path.join(folder, '{}.tif').format(i), dna.draw_particles())
            else:
                for channel, kymo in dna.draw_particles().items():
                    skimage.io.imsave(os.path.join(folder, '{}.{}.tif').format(i, channel), kymo)


@extensions.append
class DNA(dict, cats.utils.pickle_save):
    """A DNA molecule recorded across different channels.

    Parameters:
    ------------
    channels: several types
        The channels in which the DNA molecule was recorded

    Parameters for channels
    ------------------------
    name: dict key compatible
        The name of the channel
    images: cats.images
        The images in which the DNA molecule was recorded
    beginning: numeric 2-tuple
        The coordinates of the beginning (barrier) of the DNA molecule, as (x, y) coordinates.
    end: numeric 2-tuple
        The coordinates of the end (pedestal) of the DNA molecule, as (x, y) coordinates.

    Alternatively, a `DNAChannel` object can be given instead of the `images`, `beginning` and `end` parameters.

    Channels can be passed to the object as keyword arguments or list/tuples, just like it would be done for a dictionary. For example:
    >>> d = DNA(channel1=(images, beginning, end), channel2=(images2, (x1, y1), (x2, y2)))
    >>> d =  DNA("channel1", images, beginning, end), ("channel2", images2, (x1, y1), (x2, y2))

    If only images and coordinates are given, DNAChannel molecule will be instanciated instead. The build a DNA object with a single channel, make sure to name the channel.

    """

    def __new__(cls, *args, **kwargs):
        """Decide what object to build."""
        if len(kwargs) == 0 and len(args) == 3 and type(args[0]) is cats.images.Images:
            return DNAChannel(*args)
        else:
            return super().__new__(cls, args, kwargs)

    def __init__(self, *args, **kwargs):
        """Instanciate a DNA molecule."""
        if len(args) > 0 and type(args[0]) == str:
            args = args,
        for name, images, beginning, end in args:
            self[name] = DNAChannel(images, beginning, end)
        for name, params in kwargs.items():
            self[name] = DNAChannel(*params)

        # That would be nice but would lead to some misinterpretations when the user adds some attributes... Should implement otherwise.
        # self.__dict__ = self

    def __getattr__(self, attr):
        """Return attributes from the channels if not found in the object."""
        if len(self) > 0:
            # Get the channel if that's what the user wants!
            if attr in self.keys():
                return self[attr]

            # Get attribute from the underlying channels
            elif hasattr(DNAChannel, attr):
                if type(getattr(DNAChannel, attr)) == property:
                    return {name: getattr(channel, attr) for name, channel in self.items()}
                else:
                    def all_attr(*args, **kwargs):
                        return {name: getattr(channel, attr)(*args, **kwargs) for name, channel in self.items()}
                    return all_attr
            elif any([hasattr(v, attr) for v in self.values()]):
                return {name: getattr(channel, attr, None) for name, channel in self.items()}
        super().__getattribute__(attr)

    def __dir__(self):
        """Return all attributes, including those of the channels."""
        d = super().__dir__()
        [d.extend(v.__dir__()) or d.extend(v.__dict__.keys()) for v in self.values()]
        return list(set(d))

    @property
    def channels(self):
        """The channels in the DNA molecule as (name, channel)."""
        return self.items()

    @property
    def channel_names(self):
        """The names of the channels in the DNA molecule."""
        return tuple(self.keys())

    @property
    def length(self):
        """The length of the DNA molecule, in pixels."""
        lengths = self.__getattr__('length')
        one_length = np.unique(tuple(lengths.values()))
        if len(one_length) == 1:
            return one_length[0]
        else:
            return lengths

    def add_channel(self, name, images, beginning=None, end=None):
        """Add a channel to the DNA molecule.

        Parameters:
        -----------
        name: str
            The name of the channel
        images: cats.images
            The images in which the DNA molecule was recorded
        beginning: numeric 2-tuple
            The coordinates of the beginning (barrier) of the DNA molecule, as (x, y) coordinates.
        end: numeric 2-tuple
            The coordinates of the end (pedestal) of the DNA molecule, as (x, y) coordinates.

        The parameters `images`, `beginning` and `end` can be replaced by a DNAChannel object.

        """
        if type(images) == DNAChannel:
            self[name] = images
        else:
            self[name] = DNAChannel(images, beginning, end)

    def merge_kymograms(self, channels=None, colors=None, f=None):
        """Merge kymograms from the given channels.

        Parameters:
        -----------
        channels: list of str, None
            The channels to use for the merge, in the order wanted. If None, will use all channels, sorted alphabetically.
        colors: list of matplotlib.colors
            The color to use for each channel. If None, will use the colors from the default color scheme `cats.colors.scheme`
        f: path
            The file in which to save the image. If None, only returns the merge.

        Returns:
        --------
        kymogram: np.array
            The 8-bit RGB image of the merge.

        Notes:
        -----
        Kymograms in all channels must be of the same shape for this function to work.

        """
        if channels is None:
            channels = sorted(self.channel_names)
        if colors is None:
            colors = cats.colors.scheme[:len(channels)]
        a = np.array([self[channel].kymogram for channel in channels])
        merge = pims.display.to_rgb(a, colors=colors)
        if f is not None:
            skimage.io.imsave(f, merge)
        return merge.view(cats.kymograms.Kymogram)


@extensions.append
class DNAChannel(cats.utils.pickle_save):
    """A single-channel DNA molecule.

    Parameters:
    -----------
    images: cats.images
        The images in which the DNA molecule was recorded
    beginning: numeric 2-tuple
        The coordinates of the beginning (barrier) of the DNA molecule, as (x, y) coordinates.
    end: numeric 2-tuple
        The coordinates of the end (pedestal) of the DNA molecule, as (x, y) coordinates.

    """

    def __init__(self, images, beginning, end):
        """Instanciate the object."""
        self.images, self.beginning, self.end = images, beginning, end
        self.pixels = skimage.draw.line(beginning[1], beginning[0], end[1], end[0])
        self.roi_spacing = 3

    @property
    def kymogram(self):
        """Extract the kymogram of the dna molecule from the dataset."""
        if not hasattr(self, '_kymo'):
            self._kymo = cats.kymograms.Kymogram.from_slice(self.pixels, self.images)
        return self._kymo

    @kymogram.setter
    def kymogram(self, k):
        self._kymo = k

    @property
    def roi(self):
        """Return a section of the images centered around the DNA molecule, with DNAChannel.roi_spacing above and below the pixels of the DNA."""
        if not hasattr(self, '_roi'):
            x0, y0 = self.beginning
            x1, y1 = self.end
            y_max, x_max = self.images.shape[1:]
            # Consider the orientation of the data
            x_direction = -1 if x0 > x1 else 1
            if y0 > y1:
                y_slice = slice(min(y_max, y0 + self.roi_spacing), max(0, y1 - self.roi_spacing), -1)
            else:
                y_slice = slice(max(0, y0 - self.roi_spacing), min(y_max, y1 + self.roi_spacing), 1)
            x_slice = slice(max(0, x0), min(x1, x_max), x_direction)
            self._roi = self.images[:, y_slice, x_slice]
        return self._roi

    @roi.setter
    def roi(self, r):
        self._roi = r

    @property
    def length(self):
        """The length of the DNA molecule, in pixels."""
        return len(self.pixels[0])

    def track(self, *args, **kwargs):
        """Track particles from the DNA's region of interest.

        Parameters:
        -----------
        any keyword argument to be passed to the `cats.detect.track` function.

        """
        self.particles = cats.detect.track(self.roi, *args, **kwargs)
        return self.particles

    def draw_particles(self):
        """Draw the detected particles onto the kymogram , each one with a different color."""
        kymo = self.kymogram.as_rgb()
        if hasattr(self, 'particles') and self.particles.number > 0:
            for i, particle in enumerate(self.particles):
                color = cats.colors.random()
                xs, frames = particle['x'].values.astype(int), particle['frame'].values.astype(int)
                kymo[xs, frames] = color
        return kymo

    def as_multichannel(self, name):
        """Return this object as a multichannel DNA object.

        Parameter:
        -----------
        name: str
            the name of the channel this object will become
        """
        d = DNA()
        d.__dict__ = self.__dict__.copy()  # Copy potential user attributes.
        d.add_channel(name, self.images, self.beginning, self.end)
        return d

    def __repr__(self):
        """Return the coordinates of the DNA molecule."""
        return "DNA channel starting at {} and ending at {} in  {}".format(self.beginning, self.end, self.images)


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
