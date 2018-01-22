# -*- coding: utf8 -*-
"""Draw and assemble kymograms."""
from __future__ import absolute_import, division, print_function

import numpy as np
import skimage
import skimage.draw
import math
import pims.display
from cats import extensions


@extensions.append
class Kymogram(np.ndarray):
    """Create and handle a kymogram.

    A kymogram is a displacement plot built with raw data. It is a projection of a given axis along time.

    `Kymogram` objects are nothing more than numpy arrays with special methods. They can be built using any input acceptable to the `numpy.array` function, or using the `from_coordinates` static method, as follow:
    ```python
    >>> images = cats.Images('/path/to/images/*.tif')
    >>> kymogram = Kymogram.from_coordinates((12, 12), (120, 58), images)
    >>> type(kymogram)
    cats.kymograms.Kymogram
    >>> other_kymogram = Kymogram(np.ones((10, 120)))
    >>> type(other_kymogram)
    cats.kymograms.Kymogram
    ```

    """

    @staticmethod
    def from_coordinates(beginning, end, images):
        """Build a kymogram from the given coordinates.

        Parameters:
        -----------
        beginning: 2-tuple of int
            the coordinates of the initial pixel of the kymogram as (x, y)
        end: 2-tuple of int
            the coordinates of the last pixel of the kymogram as (x, y). This last pixel will be included into the kymogram.
        images: cats.images.Images
            the images in which these coordinates are to be taken.

        """
        slice_ = skimage.draw.line(beginning[1], beginning[0], end[1], end[0])
        a = [image[slice_] for image in images]
        return np.array(a, dtype=images.dtype).T.view(Kymogram)

    def from_slice(s, images):
        """Build a kymogram from the given slice.

        Parameters:
        -----------
        s: lists of 2D coordinates
            an advanced index returning the pixels of the kymogram. For example, the output of `skimage.draw.line`.
        images: cats.images.Images
            the images in which this slice is to be taken.

        """
        a = [image[s] for image in images]
        return np.array(a, dtype=images.dtype).T.view(Kymogram)

    def __new__(cls, *args, **kwargs):
        """Create a new instance of Kymogram."""
        if len(args) == 0 and len(kwargs) == 0:
            args = [[]]
        return np.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        """Save the attributes."""
        self.display_width = 512

    def as_rgb(self, rgb=(1, 1, 1)):
        """Return the kymogram as an RGB array.

        Parameters:
        -----------
        rgb: `matplotlib.colors.ColorConverter`-compatible color
            The RGB color of the signal

        """
        return pims.display.to_rgb(self, rgb)

    def save(self, f):
        """Save the kymogram to the given file.

        Parameters:
        ----------
        f: str
            The file to save the kymogram in.

        """
        skimage.io.imsave(f, self)

    def save_as_rgb(self, f, rgb=(1, 1, 1)):
        """Save the kymogram to the given file as an 8-bit RGB image, instead of greyscale.

        Parameters:
        ----------
        f: str
            The file to save the kymogram in.
        rgb: `matplotlib.colors.ColorConverter`-compatible color
            The RGB color to give to the signal

        """
        skimage.io.imsave(f, self.as_rgb(rgb))

    def _repr_png_(self):
        """Show the kymogram in Jupyter."""
        if len(self.shape) < 3:
            i = self.as_rgb()
        else:
            i = self
        return pims.display._as_png(i, width=self.display_width)


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
