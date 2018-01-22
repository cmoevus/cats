# -*- coding: utf8 -*-
"""Draw and assemble kymograms."""
from __future__ import absolute_import, division, print_function

import numpy as np
import skimage
import skimage.draw
import numbers
import math

import cats.colors
import cats.images
from cats import extensions


@extensions.append
class Kymogram:
    """Useful for kymograms.

    Parameters:
    -----------
    dataset: cat.ROI object
        the dataset to extract the kymogram from
    pt1, pt2: tuples of ints
        the positions (x, y) of the start and end points of the kymogram in the dataset. Note that the lenght must be identical in all channels.
    name: string
        name of the content in the channel (optional)
    One can list as many (dataset, pt1, pt2, name) elements as they have channels, by putting them between parantheses, with an optional name to the channel.

    """

    def __init__(self, *channels):
        """Set up the kymogram."""
        if type(channels[0]) is cats.Images:  # Single channel
            channels = [channels]
        self.pixels = None  # The kymogram
        self.raw = None  # The raw, unmodified kymogram
        self.datasets, self.points, self.names, self.pixel_positions, self.pixels = [], [], [], [], None
        for i, c in enumerate(channels):
            self.datasets.append(c[0])
            self.points.append(((int(round(c[1][0][0], 0)), int(round(c[1][0][1], 0))),
                               (int(round(c[1][1][0], 0)), int(round(c[1][1][1], 0)))))
            self.pixel_positions.append(skimage.draw.line(self.points[i][0][1], self.points[i][0][0], self.points[i][1][1], self.points[i][1][0]))
            if len(c) > 2:
                self.names.append(c[2])
            else:
                self.names.append(None)

    @property
    def shape(self):
        """Return the shape of the kymogram, as (number of channels, DNA length, number of frames)."""
        if self.raw is not None:
            return self.raw.shape
        else:
            length = min([len(d) for d in self.datasets])
            return len(self.datasets), len(self.pixel_positions[0][0]), length

    def build_empty(self):
        """Build the shape of the kymogram, without any data in it."""
        frames = min([len(d) for d in self.datasets])
        self.raw = np.zeros((len(self.datasets), len(self.pixel_positions[0][0]), frames), dtype=np.uint16)

    def build(self):
        """Extract the kymograms from the datasets."""
        self.build_empty()  # Build the structure
        length = self.shape[2]
        for i, dataset in enumerate(self.datasets):
            dataset = self.datasets[i]
            for j in range(length):
                frame = dataset[j]
                self.raw[i, :, j] = frame[self.pixel_positions[i]]
        self.pixels = self.raw.copy()
        return self.pixels

    def get(self, channels=False):
        """Extract the kymogram from the dataset.

        Parameters:
        -----------
        channels: list of numbers
            the channels to use for the kymogram. If False, uses all channels.

        Returns:
        --------
        Kymogram in the shape (channel, position, frame)

        """
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]
        if self.pixels is None:
            self.build()
        return np.squeeze(self.pixels[channels])

    def reset(self):
        """Reset the scaling and modifications to the kymogram."""
        self.pixels = self.raw.copy()

    def rescale(self, scales='image', channels=False):
        """Rescale the kymogram to the desired values.

        Parameters:
        -----------
        scales: list of 2-tuples
            The min and max value for each channel. If same for all channels, one can input a 2-tuple only.

        """
        # Check the input
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]
        if isinstance(scales[0], numbers.Number) or isinstance(scales, str):
            scales = [scales for c in channels]

        # Do
        for c in channels:
            self.pixels[c] = skimage.exposure.rescale_intensity(self.pixels[c], in_range=scales[c])
        return self.pixels
    #
    # def __repr__(self):
    #     """Representation of the kymogram."""
    #     return self.pixels

    def draw(self, colors=None, channels=False):
        """Draw the kymogram as an RGB image.

        Parameters:
        -----------
        colors: list of 3-tuples
            the RGB colors to use for each channel. If None, all channels will be white. The range for the colors goes from 0 to 1, like for matplotlib, and not 0, to 255 like usual RGB.
        channels: list of numbers
            the channels to use for the kymogram. If False, uses all channels.

        """
        if self.pixels is None:
            self.get()
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]
        if colors is None:
            colors = [(1, 1, 1) for i in channels]
        images = [cats.images.color_grayscale_image(self.pixels[c], colors[i]) for i, c in enumerate(channels)]
        return cats.images.blend_rgb_images(*images) if len(images) > 1 else images[0]


def import_handtracking(track_file):
    """Transform a file of tracks manually picked from kymograms into a list.

    Parameters:
    -----------
    track_file: str
        path to the file containing the tracks

    File format:
    ------------
    # kymogram number 0
    frame 1 track 1  (kymogram's x)  x 1 track 1 (kymogram's y)
    frame 2 track 1 (kymogram's x)   x 2 track 1 (kymogram's y)
    ...

    frame 1 track 2  (kymogram's x)  x 1 track 2 (kymogram's y)
    frame 2 track 2 (kymogram's x)   x 2 track 2 (kymogram's y)
    frame 3 track 2 (kymogram's x)   x 3 track 2 (kymogram's y)
    ...

    ...

    # kymogram number 1
    ...

    Hence, it is:
    - Define kymogram number by starting the line with a `#`.
    - Add features location underneath in the format `frame` -tab- `x`
    - Separate tracks within the same kymogram with an empty line
    - You can have as many kymograms, as many tracks per kymogram and as many features per track that you'd like.

    Notes:
    ------
    - The kymogram id has to be an int. The number should match with the DNA molecule number of the `dna.csv_to_coordinates` function, if you built the kymogram this way, to ease your work with `particles.from_kymogram_handtracks`.

    Returns:
    --------
    tracks: dict
        A dictionary of tracks as follow:
        {kymo_id_0: [(frame0, x0), (frame1, x1), ...], [(frame0, x0), ...], ...],
        kymo_id_x: [(frame0, x0), (frame1, x1), ...], [(frame0, x0), ...], ...],
        ...}

    """
    with open(track_file, 'r') as f:
        raw_tracks = f.read()

    tracks = dict()
    for kymogram in raw_tracks.split('#'):
        kymo_tracks = kymogram.split('\n\n')
        split = kymo_tracks[0].partition('\n')
        kymo_id, kymo_tracks[0] = split[0].strip(), split[2]
        if kymo_id != '':
            kymo_id = int(kymo_id)
            tracks[kymo_id] = list()
            for track in kymo_tracks:
                t = list()
                for feature in track.split('\n'):
                    positions = feature.split('\t')
                    if len(positions) > 1:
                        t.append((int(math.floor(float(positions[0]))), float(positions[1])))
                if len(track) > 2:
                    tracks[kymo_id].append(t)

    return tracks
